import argparse
import gc
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[3]
FEAT_DIR = ROOT / "tools" / "habit_audit" / "poc" / "features"
FEAT_NAMES = ["std", "white", "edge", "is_gutter", "gutter_width", "content_density", "dist_top"]
TOL_NORMAL = 100
TOL_HARD = 200

PARAMS = {
    "min_segment": 1500,     # อย่าตัดถี่เกิน (segment ด้านล่างของจุดตัด)
    "max_segment": 7000,     # ยาวเกิน -> บังคับตัด (จาก p95 ~6,488)
    "score_thresh": 0.5,
}


def load_episode(feats_path):
    z = np.load(feats_path)
    return z


def build_train_rows(meta, train_series, max_per_ep=None):
    X, y = [], []
    for e in meta:
        if e["series"] not in train_series:
            continue
        z = np.load(e["feats_path"])
        feats = z["feats"]
        idx = np.concatenate([z["pos"], z["neg"]])
        lbl = np.concatenate([np.ones(len(z["pos"])), np.zeros(len(z["neg"]))])
        if max_per_ep and len(idx) > max_per_ep:
            keep = np.random.default_rng(0).choice(len(idx), max_per_ep, replace=False)
            idx, lbl = idx[keep], lbl[keep]
        X.append(feats[idx])
        y.append(lbl)
    return np.vstack(X).astype(np.float32), np.concatenate(y).astype(np.int8)


def sim_cut_points(feats, model):
    """walk the strip top-down, scan for cut points like the real app"""
    return run_simulation(feats, model, PARAMS)


def run_simulation(feats, model, p):
    """Greedy sequential scan เหมือนโปรแกรมจริง:
    - เดินจากบนลงล่าง สะสม segment height
    - ในหน้าต่าง [prev+min_seg, prev+max_seg] หา gutter ที่ score ดี -> ตัด (snap offset)
    - ถ้าไม่เจอ gutter ที่ score พอใน window (content ต่อเนื่อง) -> บังคับตัดที่จุด content_density ต่ำสุด
    """
    H = feats.shape[0]
    probs = model.predict_proba(feats)[:, 1]
    gwidth = feats[:, FEAT_NAMES.index("gutter_width")]
    density = feats[:, FEAT_NAMES.index("content_density")]

    cuts = []
    prev = 0
    while prev + p["min_segment"] < H:
        lo = prev + p["min_segment"]
        hi = min(prev + p["max_segment"], H - 1)
        if lo > hi:
            break
        band = slice(lo, hi + 1)
        g = gwidth[band] >= 5
        scores = probs[band].copy()
        scores[~g] = -1.0
        best_local = int(np.argmax(scores))
        best_y = lo + best_local
        if scores[best_local] >= p["score_thresh"]:
            cuts.append(int(best_y))
            prev = best_y
        else:
            best_y = lo + int(np.argmin(density[band]))
            cuts.append(int(best_y))
            prev = best_y
    return np.array(cuts, dtype=np.int32)


def evaluate_episode(pred_cuts, seams, seam_in_gutter):
    """recall@tol: จุดตัดจริงที่หาเจอภายใน tolerance (แยก normal/hard)"""
    if len(seams) == 0:
        return None
    hit = 0
    used = set()
    for i, s in enumerate(seams):
        tol = TOL_NORMAL if seam_in_gutter[i] else TOL_HARD
        d = np.abs(pred_cuts - s)
        best = int(np.argmin(d)) if len(d) else -1
        if best >= 0 and d[best] <= tol and best not in used:
            hit += 1
            used.add(best)
    recall = hit / len(seams)
    # precision: predicted cuts that match a real seam
    if len(pred_cuts):
        matched = 0
        for pc in pred_cuts:
            d = np.abs(seams - pc)
            if d.min() <= max(TOL_NORMAL, TOL_HARD):
                matched += 1
        precision = matched / len(pred_cuts)
    else:
        precision = 0.0
    return recall, precision, len(pred_cuts), len(seams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", type=Path, default=FEAT_DIR)
    ap.add_argument("--max-per-ep", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    meta_path = args.feat_dir / "meta.json"
    meta = json.load(open(meta_path, encoding="utf-8"))
    series = sorted({e["series"] for e in meta})
    print(f"Episodes: {len(meta)} | Series: {series}")

    rows_all = []
    for fold_i, test_series in enumerate(series):
        train_series = [s for s in series if s != test_series]
        X, y = build_train_rows(meta, train_series, args.max_per_ep)
        print(f"\n[Fold {fold_i + 1}] test={test_series} | train rows={len(y)} "
              f"(pos={int(y.sum())} neg={int((1 - y).sum())})")
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                             random_state=0, early_stopping=True,
                                             validation_fraction=0.1)
        clf.fit(X, y)
        gc.collect()

        # evaluate on test series episodes (load one at a time = memory-safe)
        tot_r, tot_p, tot_pred, tot_real = 0.0, 0.0, 0, 0
        n_ep = 0
        for e in meta:
            if e["series"] != test_series:
                continue
            z = np.load(e["feats_path"])
            feats = z["feats"]
            seams = z["seams"].astype(np.int32)
            seg = z["seam_in_gutter"]
            pred = run_simulation(feats, clf, PARAMS)
            r = evaluate_episode(pred, seams, seg)
            if r is None:
                continue
            recall, prec, np_red, ns = r
            tot_r += recall * ns
            tot_p += prec * np_red
            tot_pred += np_red
            tot_real += ns
            n_ep += 1
            del feats, z
            gc.collect()
        print(f"  episodes={n_ep} | recall={tot_r / max(1, tot_real) * 100:.1f}% "
              f"| precision={tot_p / max(1, tot_pred) * 100:.1f}% "
              f"(pred={tot_pred} real={tot_real})")
        rows_all.append({"test_series": test_series,
                         "recall": tot_r / max(1, tot_real),
                         "precision": tot_p / max(1, tot_pred),
                         "pred": tot_pred, "real": tot_real})

    avg_recall = np.mean([r["recall"] for r in rows_all]) * 100
    avg_prec = np.mean([r["precision"] for r in rows_all]) * 100
    print(f"\n==== AVERAGE (LOSO) ====")
    print(f"recall = {avg_recall:.1f}% | precision = {avg_prec:.1f}%")

    out = args.feat_dir / "poc_result.json"
    json.dump({"params": PARAMS, "folds": rows_all,
               "avg_recall": avg_recall, "avg_precision": avg_prec},
              open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()
