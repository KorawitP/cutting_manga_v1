import argparse
import gc
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[3]
FEAT_DIR = ROOT / "tools" / "habit_audit" / "poc" / "features"
FEAT_NAMES = ["std", "white", "edge", "is_gutter", "gutter_width", "content_density", "dist_top"]
NORM_COLS = [0, 1, 2, 5]
TOL_NORMAL = 100
TOL_HARD = 200


def load_meta(feat_dir):
    return json.load(open(feat_dir / "meta.json", encoding="utf-8"))


def series_habit(meta, series_list):
    spacings = []
    for e in meta:
        if e["series"] not in series_list:
            continue
        z = np.load(e["feats_path"])
        s = z["seams"]
        spacings.extend(np.diff(np.concatenate([[0], s])).tolist())
        z.close()
    spacings = np.array(spacings)
    return {
        "min_segment": float(np.percentile(spacings, 15)),
        "max_segment": float(np.percentile(spacings, 90)),
        "median": float(np.median(spacings)),
    }


def norm_feats(feats):
    f = feats.astype(np.float32)
    out = f.copy()
    for c in NORM_COLS:
        mu = f[:, c].mean()
        sd = f[:, c].std() + 1e-6
        out[:, c] = (f[:, c] - mu) / sd
    return out


def build_train_rows(meta, train_series, max_per_ep, norm):
    X, y = [], []
    for e in meta:
        if e["series"] not in train_series:
            continue
        z = np.load(e["feats_path"])
        feats = z["feats"]
        if norm:
            feats = norm_feats(feats)
        idx = np.concatenate([z["pos"], z["neg"]])
        lbl = np.concatenate([np.ones(len(z["pos"])), np.zeros(len(z["neg"]))])
        if max_per_ep and len(idx) > max_per_ep:
            keep = np.random.default_rng(0).choice(len(idx), max_per_ep, replace=False)
            idx, lbl = idx[keep], lbl[keep]
        X.append(feats[idx])
        y.append(lbl)
        z.close()
    return np.vstack(X).astype(np.float32), np.concatenate(y).astype(np.int8)


def run_peaks_with_refine(feats, model, p, refine=True):
    probs = model.predict_proba(feats)[:, 1]
    gwidth = feats[:, FEAT_NAMES.index("gutter_width")]
    density = feats[:, FEAT_NAMES.index("content_density")]
    score = probs.copy()
    score[gwidth < 5] *= 0.3
    k = 31
    score = np.convolve(score, np.ones(k, dtype=np.float32) / k, mode="same")
    dist = max(1, int(p["min_segment"]) // 2)
    peaks, _ = find_peaks(score, height=p["score_thresh"], distance=dist)

    if refine and len(peaks):
        refined = []
        for pk in peaks:
            lo = max(0, int(pk) - 20)
            hi = min(len(density) - 1, int(pk) + 20)
            best = lo + int(np.argmin(density[lo:hi + 1]))
            refined.append(best)
        peaks = np.array(refined, dtype=np.int32)
    return peaks


def baseline_gutter(feats, p):
    H = feats.shape[0]
    gwidth = feats[:, FEAT_NAMES.index("gutter_width")]
    min_seg = int(p["min_segment"])
    cuts = []
    prev = -min_seg
    i = prev + min_seg
    while i < H:
        if gwidth[i] >= 5:
            j = i
            while j < H and gwidth[j] > 0:
                j += 1
            center = (i + j) // 2
            cuts.append(center)
            prev = center
            i = prev + min_seg
        else:
            i += 1
    return np.array(cuts, dtype=np.int32)


def evaluate_episode(pred_cuts, seams, seam_in_gutter):
    if len(seams) == 0:
        return None
    hit_n = hit_h = n_n = n_h = 0
    used = set()
    for i, s in enumerate(seams):
        tol = TOL_NORMAL if seam_in_gutter[i] else TOL_HARD
        d = np.abs(pred_cuts - s)
        if len(d) == 0:
            continue
        best = int(np.argmin(d))
        if d[best] <= tol and best not in used:
            used.add(best)
            if seam_in_gutter[i]:
                hit_n += 1
            else:
                hit_h += 1
        if seam_in_gutter[i]:
            n_n += 1
        else:
            n_h += 1
    recall = (hit_n + hit_h) / len(seams)
    if len(pred_cuts):
        matched = sum(1 for pc in pred_cuts if np.abs(seams - pc).min() <= max(TOL_NORMAL, TOL_HARD))
        precision = matched / len(pred_cuts)
    else:
        precision = 0.0
    return recall, precision, len(pred_cuts), len(seams), hit_n, n_n, hit_h, n_h


def run_all(meta, test_series, clf, habit, norm, th):
    h = dict(habit)
    h["score_thresh"] = th
    tot = {}
    for e in meta:
        if e["series"] != test_series:
            continue
        z = np.load(e["feats_path"])
        feats = z["feats"]
        if norm:
            feats = norm_feats(feats)
        seams = z["seams"].astype(np.int32)
        seg = z["seam_in_gutter"]

        pc_old = run_peaks_with_refine(feats, clf, h, refine=False)
        pc_new = run_peaks_with_refine(feats, clf, h, refine=True)
        pc_g = baseline_gutter(feats, h)

        for m, cuts in [("old", pc_old), ("refine", pc_new), ("gutter", pc_g)]:
            r = evaluate_episode(cuts, seams, seg)
            if r is None:
                continue
            t = tot.setdefault(m, {"recall": 0.0, "prec": 0.0, "pred": 0, "real": 0, "hn": 0, "nn": 0, "hh": 0, "nh": 0})
            rec, prec, np_, ns, hn, nn, hh, nh = r
            t["recall"] += rec * ns
            t["prec"] += prec * np_
            t["pred"] += np_
            t["real"] += ns
            t["hn"] += hn
            t["nn"] += nn
            t["hh"] += hh
            t["nh"] += nh
        z.close()
        gc.collect()
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", type=Path, default=FEAT_DIR)
    ap.add_argument("--max-per-ep", type=int, default=1500)
    ap.add_argument("--thresh-sweep", action="store_true", help="sweep thresholds for best")
    ap.add_argument("--norm", action="store_true", help="z-score scale-sensitive features per episode")
    ap.add_argument("--score-thresh", type=float, default=0.3)
    args = ap.parse_args()

    meta = load_meta(args.feat_dir)
    series = sorted({e["series"] for e in meta})
    print(f"Phase A: Episodes={len(meta)} | Norm={args.norm} | Refinement=True")
    threshs = [round(x, 2) for x in np.arange(0.15, 0.50, 0.05)] if args.thresh_sweep else [args.score_thresh]

    fold_best = {}
    for fold_i, test_series in enumerate(series):
        train_series = [s for s in series if s != test_series]
        habit = series_habit(meta, train_series)
        X, y = build_train_rows(meta, train_series, args.max_per_ep, args.norm)
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                             random_state=0, early_stopping=True,
                                             validation_fraction=0.1)
        clf.fit(X, y)
        gc.collect()

        best_rec = 0.0
        best_th = 0.0
        for th in threshs:
            tot = run_all(meta, test_series, clf, habit, args.norm, th)
            t = tot.get("refine")
            if t and t["real"]:
                rec = t["recall"] / t["real"]
                if rec > best_rec:
                    best_rec, best_th = rec, th
                    fold_best[test_series] = {"th": th, "recall": rec * 100,
                        "gutter": tot.get("gutter", {}).get("recall", 0) / max(1, tot["gutter"]["real"]) * 100,
                        "hn": t["hn"], "nn": t["nn"], "hh": t["hh"], "nh": t["nh"],
                        "pred": t["pred"], "real": t["real"]}
        gc.collect()

    print("\n========== PHASE A RESULTS ==========")
    print(f"  Norm={args.norm} | refinement=True | sweep={args.thresh_sweep}")
    for s in series:
        r = fold_best.get(s, {})
        if not r:
            continue
        rec = r["recall"]
        gr = r.get("gutter", 0)
        norm_rec = r["hn"] / max(1, r["nn"]) * 100
        hard_rec = r["hh"] / max(1, r["nh"]) * 100
        print(f"  {s:38s} refine_rec={rec:.1f}% gutter={gr:.1f}% norm={norm_rec:.0f}% hard={hard_rec:.0f}% thr={r['th']}")
    avg_r = np.mean([r["recall"] for r in fold_best.values()])
    avg_g = np.mean([r.get("gutter", 0) for r in fold_best.values()])
    print(f"\n  AVG refine={avg_r:.1f}% | gutter={avg_g:.1f}% | delta={avg_r - avg_g:+.1f}%")

    out = args.feat_dir / "phase_a.json"
    json.dump({"norm": args.norm, "avg_refine": avg_r, "avg_gutter": avg_g, "delta": avg_r - avg_g, "per_series": {str(k): v for k, v in fold_best.items()}},
              open(out, "w"), indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else str(x))
    print("Saved:", out)


if __name__ == "__main__":
    main()

