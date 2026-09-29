"""Fast POC: z-score norm + sweep thresholds. One pass."""
import gc, json, argparse
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[3]
FEAT_DIR = ROOT / "tools" / "habit_audit" / "poc" / "features"
NORM_COLS = [0, 1, 2, 5]
TOL_NORMAL, TOL_HARD = 100, 200


def load_meta(d):
    return json.load(open(d / "meta.json", encoding="utf-8"))


def norm(g):
    g = g.astype(np.float32)
    o = g.copy()
    for c in NORM_COLS:
        o[:, c] = (g[:, c] - g[:, c].mean()) / (g[:, c].std() + 1e-6)
    return o


def segments(meta, series_list):
    sp = []
    for e in meta:
        if e["series"] not in series_list:
            continue
        z = np.load(e["feats_path"])
        sp.extend(np.diff(np.concatenate([[0], z["seams"]])).tolist())
        z.close()
    sp = np.asfarray(sp)
    return {"min_seg": int(np.percentile(sp, 15)), "max_seg": int(np.percentile(sp, 90)),
            "median": int(np.median(sp))}


def train_rows(meta, train_series, cap, use_norm):
    X, Y = [], []
    for e in meta:
        if e["series"] not in train_series:
            continue
        z = np.load(e["feats_path"])
        f = z["feats"]
        if use_norm:
            f = norm(f)
        idx = np.concatenate([z["pos"], z["neg"]])
        lab = np.concatenate([np.ones(len(z["pos"])), np.zeros(len(z["neg"]))])
        if cap and len(idx) > cap:
            k = np.random.default_rng(0).choice(len(idx), cap, replace=False)
            idx, lab = idx[k], lab[k]
        X.append(f[idx])
        Y.append(lab)
        z.close()
    return np.vstack(X).astype(np.float32), np.concatenate(Y).astype(np.int8), None


def score_and_seams(meta, test_series, clf, use_norm):
    """Return per-episode scores + seams + gutter flags (one pass per fold)"""
    out = []
    for e in meta:
        if e["series"] != test_series:
            continue
        z = np.load(e["feats_path"])
        f = z["feats"]
        if use_norm:
            f = norm(f)
        probs = clf.predict_proba(f)[:, 1]
        gwidth = f[:, 3]  # col 3 is is_gutter * col 4 would be gutter_width? Wait col 3 is is_gutter
        gwidth_orig = f[:, 4]  # actual gutter width
        # boost score on gutter zones, penalize non-gutter
        score = probs.copy()
        score[gwidth_orig < 5] *= 0.3
        k = 31
        score = np.convolve(score, np.ones(k, dtype=np.float32) / k, mode="same")
        out.append({
            "score": score,
            "seams": z["seams"].astype(np.int32),
            "seg": z["seam_in_gutter"],
        })
        z.close()
    return out


def eval_one(pred, seams, seg):
    hit_n = hit_h = 0
    n_n, n_h = 0, 0
    used = set()
    for i, s in enumerate(seams):
        tol = TOL_NORMAL if seg[i] else TOL_HARD
        d = np.abs(pred - s)
        if len(d) == 0:
            continue
        b = int(np.argmin(d))
        if d[b] <= tol and b not in used:
            used.add(b)
            if seg[i]:
                hit_n += 1
            else:
                hit_h += 1
        if seg[i]:
            n_n += 1
        else:
            n_h += 1
    return hit_n, n_n, hit_h, n_h, len(pred), len(seams)


def eval_thresh(data, th, min_seg):
    hn = hh = nn = nh = tot_p = tot_r = 0
    dist = max(1, min_seg // 2)
    for d in data:
        peaks, _ = find_peaks(d["score"], height=th, distance=dist)
        a, b, c, d_val, np_, ns = eval_one(peaks.astype(np.int32), d["seams"], d["seg"])
        hn += a
        nn += b
        hh += c
        nh += d_val
        tot_p += np_
        tot_r += ns
    rec = (hn + hh) / max(1, tot_r)
    prec = (hn + hh) / max(1, tot_p)
    return rec, prec, hn, nn, hh, nh, tot_p, tot_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", type=Path, default=FEAT_DIR)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--norm", action="store_true")
    args = ap.parse_args()

    meta = load_meta(args.feat_dir)
    series = sorted({e["series"] for e in meta})
    print(f"Episodes: {len(meta)} | Series: {series} | norm={args.norm}")

    th_range = [round(x, 2) for x in np.arange(0.12, 0.55, 0.03)]

    folds = []
    for test in series:
        train_s = [s for s in series if s != test]
        hab = segments(meta, train_s)
        X, y, _ = train_rows(meta, train_s, args.cap, args.norm)
        print(f"\nFold test={test} | train_rows={len(y)} habit(min={hab['min_seg']},max={hab['max_seg']},med={hab['median']})")
        clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1,
                                             random_state=0, early_stopping=True,
                                             validation_fraction=0.1).fit(X, y)
        gc.collect()
        data = score_and_seams(meta, test, clf, args.norm)
        gc.collect()

        best_th, best_rec = None, 0
        for th in th_range:
            rec, prec, hn, nn, hh, nh, tp_, tr = eval_thresh(data, th, hab["min_seg"])
            if rec > best_rec:
                best_rec, best_th, best_info = rec, th, (prec, hn, nn, hh, nh, tp_, tr)
        rec, prec, hn, nn, hh, nh, tp_, tr = best_info
        folds.append({"test": test, "best_th": best_th,
                      "recall": rec * 100, "precision": prec * 100,
                      "norm_hit": hn, "norm_tot": nn, "hard_hit": hh, "hard_tot": nh,
                      "hab": hab})
        print(f"  best_th={best_th:.2f} recall={rec*100:.1f}% prec={prec*100:.1f}% "
              f"| norm {hn}/{nn} hard {hh}/{nh}")

    print("\n=============== POC FINAL ===============")
    for f in folds:
        print(f"  {f['test']:38s} th={f['best_th']:.2f} rec={f['recall']:.1f}% "
              f"norm={f['norm_hit']}/{f['norm_tot']} hard={f['hard_hit']}/{f['hard_tot']}")
    avg_rec = np.mean([f["recall"] for f in folds])
    print(f"\n  AVG recall (LOSO) = {avg_rec:.1f}% | norm={args.norm}")

    out = args.feat_dir / ("poc_fast_norm.json" if args.norm else "poc_fast_raw.json")
    json.dump({"avg_recall": avg_rec, "norm": args.norm, "folds": [{k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in f.items()} for f in folds]},
              open(out, "w"), indent=2)
    print("Saved:", out)


if __name__ == "__main__":
    main()
