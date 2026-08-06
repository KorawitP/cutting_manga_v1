import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import eval_both as E
import joblib

split = json.load(open(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json", encoding="utf-8"))
val = split["val"]
gbm = joblib.load(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\gbm_r1.joblib")

# precompute GBM probs per episode
probs_cache = []
for e in val:
    ep = E.build_features_for_episode(Path(e["images"]))
    if ep is None:
        continue
    feats = E.norm_feats(ep["feats"])
    probs = gbm.predict_proba(feats)[:, 1]
    gw = feats[:, 4]
    probs_cache.append({"probs": probs, "gw": gw, "seams": ep["seams"],
                        "seg": ep["seg"], "density": ep["density"]})

def run_gbm_sweep(th, smooth_w):
    tot_hit, tot_real, tot_prec, tot_pred = 0, 0, 0, 0
    for c in probs_cache:
        score = c["probs"].copy()
        score[c["gw"] < 5] *= 0.3
        score = np.convolve(score, np.ones(smooth_w) / smooth_w, mode="same")
        dist = max(1, 30 // 2)
        pk, _ = find_peaks(score, height=th, distance=dist)
        peaks = E.refine_peaks(pk, c["density"], len(c["probs"]))
        r = E.eval_episode(peaks, {"seams": c["seams"], "seg": c["seg"]})
        if not r:
            continue
        rec, prec, np_, ns = r
        tot_hit += rec * ns; tot_real += ns
        tot_prec += prec * np_; tot_pred += np_
    return tot_hit / max(1, tot_real) * 100, tot_prec / max(1, tot_pred) * 100

print("GBM threshold sweep (smooth=31):")
best = (0, None, None)
for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    r, p = run_gbm_sweep(th, 31)
    print(f"  th={th:.2f} recall={r:.1f}% precision={p:.1f}%")
    if r > best[0]:
        best = (r, th, 31)
print("GBM smooth sweep (th=0.25):")
for w in [11, 21, 31, 41]:
    r, p = run_gbm_sweep(0.25, w)
    print(f"  w={w} recall={r:.1f}% precision={p:.1f}%")
    if r > best[0]:
        best = (r, 0.25, w)
print(f"\nGBM BEST: th={best[1]} smooth={best[2]} recall={best[0]:.1f}%")
