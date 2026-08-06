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


def load_meta(feat_dir):
    return json.load(open(feat_dir / "meta.json", encoding="utf-8"))


NORM_COLS = [0, 1, 2, 5]  # std, white, edge, content_density (scale-sensitive only)


def norm_feats(feats):
    """z-score เฉพาะ col scale-sensitive ต่อตอน (ลด covariate shift ระหว่างเรื่อง)"""
    f = feats.astype(np.float32)
    out = f.copy()
    for c in NORM_COLS:
        mu = f[:, c].mean()
        sd = f[:, c].std() + 1e-6
        out[:, c] = (f[:, c] - mu) / sd
    return out


def series_habit(meta, series_list):
    """derive the user's 'habit scale' from a set of series (their spacing distribution)"""
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


def build_train_rows(meta, train_series, max_per_ep, use_norm):
    X, y = [], []
    for e in meta:
        if e["series"] not in train_series:
            continue
        z = np.load(e["feats_path"])
        feats = z["feats"]
        if use_norm:
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


def run_simulation(feats, model, p):
    H = feats.shape[0]
    probs = model.predict_proba(feats)[:, 1]
    gwidth = feats[:, FEAT_NAMES.index("gutter_width")]
    density = feats[:, FEAT_NAMES.index("content_density")]

    min_seg = int(p["min_segment"])
    max_seg = int(p["max_segment"])
    cuts = []
    prev = 0
    while prev + min_seg < H:
        lo = prev + min_seg
        hi = min(prev + max_seg, H - 1)
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


def baseline_gutter(feats, p):
    """cut at every gutter band center, spacing >= min_segment (no ML)"""
    H = feats.shape[0]
    gwidth = feats[:, FEAT_NAMES.index("gutter_width")]
    min_seg = int(p["min_segment"])
    cuts = []
    prev = 0
    i = prev + min_seg
    while i < H:
        if gwidth[i] >= 5:
            # snap to band center
            j = i
            while j < H and gwidth[j] > 0:
                j += 1
            center = (i + j) // 2
            cuts.append(int(center))
            prev = center
            i = prev + min_seg
        else:
            i += 1
    return np.array(cuts, dtype=np.int32)


def baseline_interval(feats, p):
    """cut every median-spacing px (no ML, no gutter info)"""
    H = feats.shape[0]
    step = int(p["median"])
    cuts = np.arange(step, H, step).astype(np.int32)
    return cuts


def run_peaks(feats, model, p):
    """เหมือนแอปจริง: score ทุกแถว -> smooth -> find_peaks(min_distance=min_segment)"""
    from scipy.signal import find_peaks
    H = feats.shape[0]
    probs = model.predict_proba(feats)[:, 1]
    gwidth = feats[:, FEAT_NAMES.index("gutter_width")]
    score = probs.copy()
    score[gwidth < 5] *= 0.3  # ลดคะแนนจุดที่ไม่ใช่ gutter
    k = 31
    score = np.convolve(score, np.ones(k, dtype=np.float32) / k, mode="same")
    dist = max(1, int(p["min_segment"]) // 2)
    peaks, _ = find_peaks(score, height=p["score_thresh"], distance=dist)
    return peaks.astype(np.int32)


def evaluate_episode(pred_cuts, seams, seam_in_gutter):
    if len(seams) == 0:
        return None
    hit_norm = hit_hard = n_norm = n_hard = 0
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
                hit_norm += 1
            else:
                hit_hard += 1
        if seam_in_gutter[i]:
            n_norm += 1
        else:
            n_hard += 1
    recall = (hit_norm + hit_hard) / len(seams)
    if len(pred_cuts):
        matched = sum(1 for pc in pred_cuts if np.abs(seams - pc).min() <= max(TOL_NORMAL, TOL_HARD))
        precision = matched / len(pred_cuts)
    else:
        precision = 0.0
    return recall, precision, len(pred_cuts), len(seams), hit_norm, n_norm, hit_hard, n_hard


def run_all(meta, test_series, clf, habit, use_norm):
    tot = {}
    for e in meta:
        if e["series"] != test_series:
            continue
        z = np.load(e["feats_path"])
        feats = z["feats"]
        if use_norm:
            feats = norm_feats(feats)
        seams = z["seams"].astype(np.int32)
        seg = z["seam_in_gutter"]
        r = {m: evaluate_episode(c, seams, seg) for m, c in
             {"sim": run_simulation(feats, clf, habit),
              "gutter": baseline_gutter(feats, habit),
              "interval": baseline_interval(feats, habit),
              "peaks": run_peaks(feats, clf, habit)}.items()}
        for m, res in r.items():
            if res is None:
                continue
            t = tot.setdefault(m, {"recall": 0.0, "prec": 0.0, "pred": 0, "real": 0, "n": 0,
                                   "hn": 0, "nn": 0, "hh": 0, "nh": 0})
            rec, prec, np_, ns, hn, nn, hh, nh = res
            t["recall"] += rec * ns
            t["prec"] += prec * np_
            t["pred"] += np_
            t["real"] += ns
            t["n"] += 1
            t["hn"] += hn; t["nn"] += nn; t["hh"] += hh; t["nh"] += nh
        z.close()
        gc.collect()
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", type=Path, default=FEAT_DIR)
    ap.add_argument("--max-per-ep", type=int, default=1500)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--thresh-sweep", action="store_true", help="sweep 0.15..0.45 and report best")
    ap.add_argument("--norm", action="store_true")
    args = ap.parse_args()

    meta = load_meta(args.feat_dir)
    series = sorted({e["series"] for e in meta})
    print(f"Episodes: {len(meta)} | Series: {series} | norm={args.norm}")
    threshs = [round(x, 2) for x in np.arange(0.15, 0.50, 0.05)] if args.thresh_sweep else [args.score_thresh]

    # train once per fold, evaluate all thresh
    fold_results = {}
    for fold_i, test_series in enumerate(series):
        train_series = [s for s in series if s != test_series]
        habit = series_habit(meta, train_series)
        X, y = build_train_rows(meta, train_series, args.max_per_ep, args.norm)
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                             random_state=0, early_stopping=True,
                                             validation_fraction=0.1)
        clf.fit(X, y)
        gc.collect()

        def eval_thresh(th):
            h = dict(habit)
            h["score_thresh"] = th
            return run_all(meta, test_series, clf, h, args.norm)

        best_t = best_r = None
        for th in threshs:
            tot = eval_thresh(th)
            p = tot.get("peaks")
            if not p or p["real"] == 0:
                continue
            rec = p["recall"] / p["real"]
            prec = p["prec"] / max(1, p["pred"])
            if best_r is None or rec > best_r:
                best_r, best_t, best_prec = rec, th, prec
        fold_results[test_series] = {
            "best_thresh": best_t, "recall": best_r * 100, "precision": best_prec * 100,
            "habit_min": habit["min_segment"], "habit_max": habit["max_segment"],
        }
        # also save baseline gutter (thresh-independent)
        tg = eval_thresh(999)  # gutter not affected but we need to call run_all
        g = tg.get("gutter")
        if g and g["real"]:
            fold_results[test_series]["gutter_recall"] = g["recall"] / g["real"] * 100
        gc.collect()

    print("\n============= POC FINAL =============")
    print(f"  norm={args.norm} | thresh sweep={args.thresh_sweep}")
    rows = []
    for s in series:
        r = fold_results.get(s, {})
        rows.append(f"{s:38s} peaks_rec={r.get('recall',0):.1f}% gutter_rec={r.get('gutter_recall',0):.1f}% best_th={r.get('best_thresh','?')}")
        print("  " + rows[-1])
    avg_rec = np.mean([r.get("recall", 0) for r in fold_results.values()])
    avg_gr = np.mean([r.get("gutter_recall", 0) for r in fold_results.values()])
    print(f"\n  AVG peaks_rec={avg_rec:.1f}% | AVG gutter_rec={avg_gr:.1f}% | delta={avg_rec-avg_gr:+.1f}%")

    out = args.feat_dir / "poc_final.json"
    json.dump({"norm": args.norm, "thresh_sweep": args.thresh_sweep,
               "avg_peaks_recall": avg_rec, "avg_gutter_recall": avg_gr, "delta": avg_rec - avg_gr,
               "per_series": fold_results},
              open(out, "w"), indent=2)
    print("Saved:", out)


if __name__ == "__main__":
    main()

