import argparse, gc, json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
FEAT_DIR = ROOT / "tools" / "habit_audit" / "poc" / "features"
FEAT_NAMES = ["std", "white", "edge", "is_gutter", "gutter_width", "content_density", "dist_top"]
NORM_COLS = [0, 1, 2, 5]
TOL_N, TOL_H = 100, 200


def norm_feats(f):
    out = f.astype(np.float32).copy()
    for c in NORM_COLS:
        mu = f[:, c].mean()
        sd = f[:, c].std() + 1e-6
        out[:, c] = (f[:, c] - mu) / sd
    return out


def load_meta():
    return json.load(open(FEAT_DIR / "meta.json", encoding="utf-8"))


def habit(meta, train_series):
    sp = []
    for e in meta:
        if e["series"] not in train_series:
            continue
        z = np.load(e["feats_path"])
        sp.extend(np.diff(np.concatenate([[0], z["seams"]])).tolist())
        z.close()
    sp = np.array(sp)
    return {"min_seg": float(np.percentile(sp, 15)), "max_seg": float(np.percentile(sp, 90)),
            "median": float(np.median(sp))}


def test_ep(probs, feats, seams, seg, h, th):
    gw = feats[:, FEAT_NAMES.index("gutter_width")]
    de = feats[:, FEAT_NAMES.index("content_density")]
    pk = run_peaks(probs, gw, de, h["min_seg"], th)
    return eval_ep(pk, seams, seg)


def run_peaks(probs, gwidth, density, min_seg, th):
    score = probs.copy()
    score[gwidth < 5] *= 0.3
    score = np.convolve(score, np.ones(31, dtype=np.float64) / 31, mode="same")
    dist = max(1, int(min_seg) // 2)
    peaks, _ = find_peaks(score, height=th, distance=dist)
    if len(peaks):
        ref = []
        for pk in peaks:
            lo = max(0, int(pk) - 20)
            hi = min(len(density) - 1, int(pk) + 20)
            ref.append(lo + int(np.argmin(density[lo:hi + 1])))
        peaks = np.array(ref, dtype=np.int32)
    return peaks


def eval_ep(pred, seams, seg):
    if len(seams) == 0:
        return 0.0, 0, 0, 0, 0, 0
    hit_n, hit_h, n_n, n_h = 0, 0, 0, 0
    used = set()
    for i, s in enumerate(seams):
        tol = TOL_N if seg[i] else TOL_H
        d = np.abs(pred - s)
        if len(d) == 0:
            continue
        best = int(np.argmin(d))
        if d[best] <= tol and best not in used:
            used.add(best)
            if seg[i]:
                hit_n += 1
            else:
                hit_h += 1
        if seg[i]:
            n_n += 1
        else:
            n_h += 1
    return (hit_n + hit_h) / len(seams), hit_n + hit_h, n_n, n_h, hit_n, hit_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-label", type=int, default=50, help="number of labeled cuts per test series for adaptation")
    ap.add_argument("--max-per-ep", type=int, default=1500)
    args = ap.parse_args()
    meta = load_meta()
    series = sorted({e["series"] for e in meta})

    print(f"Phase B: per-series adapt ({args.n_label} labels per test series)")
    for test_series in series:
        train_series = [s for s in series if s != test_series]
        h = habit(meta, train_series)
        # train global GBM
        X, y = [], []
        for e in meta:
            if e["series"] not in train_series:
                continue
            z = np.load(e["feats_path"])
            f = norm_feats(z["feats"])
            idx = np.concatenate([z["pos"], z["neg"]])
            lbl = np.concatenate([np.ones(len(z["pos"])), np.zeros(len(z["neg"]))])
            if len(idx) > args.max_per_ep:
                keep = np.random.default_rng(0).choice(len(idx), args.max_per_ep, replace=False)
                idx, lbl = idx[keep], lbl[keep]
            X.append(f[idx])
            y.append(lbl)
            z.close()
        X = np.vstack(X).astype(np.float32)
        y = np.concatenate(y).astype(np.int8)
        gbm = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                             random_state=0, early_stopping=True,
                                             validation_fraction=0.1).fit(X, y)
        # --- simulate: label args.n_label seams from the test series ---
        rng = np.random.default_rng(42)
        # pick labeled seams from test series (first episode)
        test_eps = [e for e in meta if e["series"] == test_series]
        adapt_X, adapt_y = [], []
        for e in test_eps[:max(1, args.n_label // 30)]:
            z = np.load(e["feats_path"])
            f = norm_feats(z["feats"])
            s = z["seams"].astype(np.int32)
            n_pick = min(len(s), args.n_label)
            picks = rng.choice(s, size=max(1, n_pick), replace=False)
            for p in picks:
                lo = max(0, p - 2)
                hi = min(f.shape[0] - 1, p + 2)
                adapt_X.append(f[lo:hi + 1].mean(axis=0))
                adapt_y.append(1)
            # negatives: random non-seam rows
            neg_idx = []
            all_rows = np.arange(f.shape[0])
            seam_mask = np.zeros(f.shape[0], dtype=bool)
            for st in s:
                seam_mask[max(0, st - 3):min(f.shape[0], st + 4)] = True
            ok = all_rows[~seam_mask]
            if len(ok) > n_pick * 2:
                neg_picks = rng.choice(ok, size=n_pick * 2, replace=False)
            else:
                neg_picks = ok
            for pg in neg_picks:
                adapt_X.append(f[pg])
                adapt_y.append(0)
            z.close()
        adapt_X = np.vstack(adapt_X).astype(np.float32)
        adapt_y = np.array(adapt_y)

        # adapter: logistic on GBM outputs → calibrated for this series
        gbm_prob = gbm.predict_proba(adapt_X)[:, 1:]
        if gbm_prob.shape[1] == 1:
            gbm_prob = gbm_prob.ravel()
        adapter = LogisticRegression(C=10, max_iter=200, random_state=0).fit(
            gbm_prob.reshape(-1, 1), adapt_y)

        # evaluate: no-adapt (auto threshold sweep) vs adapt
        tot_no = {"rec": 0, "real": 0}
        tot_ad = {"rec": 0, "real": 0}
        for e in test_eps:
            z = np.load(e["feats_path"])
            f = norm_feats(z["feats"])
            seams = z["seams"].astype(np.int32)
            seg = z["seam_in_gutter"]
            probs = gbm.predict_proba(f)[:, 1]

            # no-adapt: sweep best threshold
            best_r = 0
            for th in np.arange(0.10, 0.55, 0.05):
                rr, hits, _, _, _, _ = test_ep(probs, f, seams, seg, h, th)
                if rr > best_r:
                    best_r = rr
            tot_no["rec"] += best_r * len(seams)
            tot_no["real"] += len(seams)

            # adapt: use adapter to calibrate probabilities
            prob_adapted = adapter.predict_proba(probs.reshape(-1, 1))[:, 1]
            best_r2 = 0
            for th in np.arange(0.10, 0.55, 0.05):
                rr2, hits, _, _, _, _ = test_ep(prob_adapted, f, seams, seg, h, th)
                if rr2 > best_r2:
                    best_r2 = rr2
            tot_ad["rec"] += best_r2 * len(seams)
            tot_ad["real"] += len(seams)
            z.close()
            gc.collect()

        r_no = tot_no["rec"] / max(1, tot_no["real"]) * 100
        r_ad = tot_ad["rec"] / max(1, tot_ad["real"]) * 100
        print(f"  {test_series:38s} no_adapt={r_no:.1f}% -> adapt={r_ad:.1f}%  (delta={r_ad - r_no:+.1f}%)")


if __name__ == "__main__":
    main()
