import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import eval_both as E
import joblib

TOL = 50            # user requirement: position must be within +-50px of real seam
MIN_DIST = 30       # min separation between cuts (px)
CACHE_R3 = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r3")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")

# --- CNN / GBM score arrays (same pipeline as tuned models) ---

def cnn_score_from_cache(z):
    return np.convolve(z["sv"], np.ones(11) / 11, mode="same")

def gbm_score(ep, gbm):
    feats = E.norm_feats(ep["feats"])
    probs = gbm.predict_proba(feats)[:, 1]
    score = probs.copy()
    score[feats[:, 4] < 5] *= 0.3
    return np.convolve(score, np.ones(31) / 31, mode="same")

# --- Fixed-N selection: force EXACTLY N cuts, prefer high score, min-dist apart ---

def select_n_cuts(yc, score, N, min_dist=MIN_DIST):
    order = np.argsort(score)[::-1]
    chosen = []
    for i in order:
        y = int(yc[i])
        if all(abs(y - c) >= min_dist for c in chosen):
            chosen.append(y)
        if len(chosen) >= N:
            break
    return np.sort(np.array(chosen, dtype=np.int32)) if chosen else np.array([], dtype=np.int32)

def match_1to1(cuts, seams, tol=TOL):
    cuts = np.sort(cuts)
    matched = 0
    used = set()
    for s in seams:
        if len(cuts) == 0:
            break
        d = np.abs(cuts - s)
        best = int(np.argmin(d))
        if d[best] <= tol and best not in used:
            matched += 1
            used.add(best)
    return matched

# --- Old metric (recall, tolerance 100/200) for contrast ---

def old_recall(z, ep, th, smooth_w, is_cnn, gbm):
    if is_cnn:
        sv_sm = np.convolve(z["sv"], np.ones(smooth_w) / smooth_w, mode="same")
        dist = max(1, 30 // E.STRIDE // 2)
        pk, _ = find_peaks(sv_sm, height=th, distance=dist)
        peaks = E.refine_peaks(z["yc"][pk], z["density"], int(z["H"]))
    else:
        feats = E.norm_feats(ep["feats"])
        probs = gbm.predict_proba(feats)[:, 1]
        score = probs.copy()
        score[feats[:, 4] < 5] *= 0.3
        score = np.convolve(score, np.ones(smooth_w) / smooth_w, mode="same")
        dist = max(1, 30 // 2)
        pk, _ = find_peaks(score, height=th, distance=dist)
        peaks = E.refine_peaks(pk, ep["density"], len(probs))
    r = E.eval_episode(peaks, {"seams": z["seams"] if is_cnn else ep["seams"],
                               "seg": z["seg"] if is_cnn else ep["seg"]})
    return r  # (recall, precision, n_pred, n_seams)

def main():
    split = json.load(open(SPLIT, encoding="utf-8"))
    val = split["val"]
    gbm = joblib.load(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\gbm_r1.joblib")

    agg = defaultdict(lambda: {"cnn": {"m": 0, "r": 0}, "gbm": {"m": 0, "r": 0},
                               "cnn_old": {"hit": 0, "r": 0}, "gbm_old": {"hit": 0, "r": 0}})
    n_ep = 0
    for e in val:
        name = f"{e['series']}__{e['episode']}"
        safe = (name.replace("\\", "_").replace("/", "_").replace(":", "_")
                .replace("*", "_").replace("?", "_").replace('"', "_")
                .replace("<", "_").replace(">", "_").replace("|", "_"))
        z = np.load(CACHE_R3 / f"{safe}.npz")
        seams = np.sort(z["seams"])
        N = len(seams)
        if N == 0:
            continue
        n_ep += 1
        ep = E.build_features_for_episode(Path(e["images"]))
        if ep is None:
            continue

        # Fixed-N eval (user metric)
        sv_cnn = cnn_score_from_cache(z)
        cuts_cnn = select_n_cuts(z["yc"], sv_cnn, N)
        m_cnn = match_1to1(cuts_cnn, seams)

        sv_gbm = gbm_score(ep, gbm)
        yc_g = np.arange(len(sv_gbm), dtype=np.int32)
        cuts_gbm = select_n_cuts(yc_g, sv_gbm, N)
        m_gbm = match_1to1(cuts_gbm, seams)

        # Old-metric recall (contrast)
        rc = old_recall(z, ep, 0.3, 11, True, gbm)
        rg = old_recall(z, ep, 0.1, 31, False, gbm)

        a = agg[e["series"]]
        a["cnn"]["m"] += m_cnn; a["cnn"]["r"] += N
        a["gbm"]["m"] += m_gbm; a["gbm"]["r"] += N
        if rc:
            a["cnn_old"]["hit"] += rc[0] * N; a["cnn_old"]["r"] += N
        if rg:
            a["gbm_old"]["hit"] += rg[0] * N; a["gbm_old"]["r"] += N

    tot = {"cnn": {"m": 0, "r": 0}, "gbm": {"m": 0, "r": 0},
           "cnn_old": {"hit": 0, "r": 0}, "gbm_old": {"hit": 0, "r": 0}}
    print("=" * 80)
    print("FAIR EVAL (USER METRIC): EXACTLY SAME #cuts as real, position +-50px (1:1)")
    print("  accuracy = matched / n_real     |  old recall (tol 100/200) shown for contrast")
    print("=" * 80)
    for s in sorted(agg):
        a = agg[s]
        for m in ("cnn", "gbm"):
            tot[m]["m"] += a[m]["m"]; tot[m]["r"] += a[m]["r"]
        for m in ("cnn_old", "gbm_old"):
            tot[m]["hit"] += a[m]["hit"]; tot[m]["r"] += a[m]["r"]
        cc = a["cnn"]["m"] / a["cnn"]["r"] * 100
        gg = a["gbm"]["m"] / a["gbm"]["r"] * 100
        co = a["cnn_old"]["hit"] / a["cnn_old"]["r"] * 100 if a["cnn_old"]["r"] else 0
        go = a["gbm_old"]["hit"] / a["gbm_old"]["r"] * 100 if a["gbm_old"]["r"] else 0
        print(f"  {s:38s} CNN {cc:5.1f}%  GBM {gg:5.1f}%   | old-rec: CNN {co:5.1f}% GBM {go:5.1f}%")

    print("-" * 80)
    cc = tot["cnn"]["m"] / tot["cnn"]["r"] * 100
    gg = tot["gbm"]["m"] / tot["gbm"]["r"] * 100
    co = tot["cnn_old"]["hit"] / tot["cnn_old"]["r"] * 100
    go = tot["gbm_old"]["hit"] / tot["gbm_old"]["r"] * 100
    print(f"  TOTAL ({n_ep} eps, {tot['cnn']['r']} real cuts) | "
          f"CNN {cc:.1f}% vs GBM {gg:.1f}%")
    print(f"  old-recall contrast      | CNN {co:.1f}% vs GBM {go:.1f}%  "
          f"(tolerance 100/200, free line count)")
    print(f"  delta fixed-N = {cc - gg:+.1f}% | delta old = {co - go:+.1f}%")

if __name__ == "__main__":
    main()
