import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

import sys
sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import eval_both as E

CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r3")

def eval_threshold(eps, th, smooth_w):
    tot_hit, tot_real, tot_prec, tot_pred = 0, 0, 0, 0
    for c in eps:
        sv_sm = np.convolve(c["sv"], np.ones(smooth_w) / smooth_w, mode="same")
        dist = max(1, 30 // E.STRIDE // 2)
        pk, _ = find_peaks(sv_sm, height=th, distance=dist)
        peaks = c["yc"][pk]
        peaks = E.refine_peaks(peaks, c["density"], c["H"])
        r = E.eval_episode(peaks, {"seams": c["seams"], "seg": c["seg"]})
        if not r:
            continue
        rec, prec, np_, ns = r
        tot_hit += rec * ns
        tot_real += ns
        tot_prec += prec * np_
        tot_pred += np_
    return tot_hit / max(1, tot_real) * 100, tot_prec / max(1, tot_pred) * 100

eps = []
for p in sorted(CACHE.glob("*.npz")):
    z = np.load(p)
    eps.append({"yc": z["yc"], "sv": z["sv"], "seams": z["seams"],
                "seg": z["seg"], "density": z["density"], "H": int(z["H"])})
print(f"CNN r3 episodes={len(eps)}")
print("threshold sweep (smooth=11):")
for th in [0.2, 0.25, 0.3, 0.35, 0.4]:
    r, p = eval_threshold(eps, th, 11)
    print(f"  th={th:.2f} recall={r:.1f}% precision={p:.1f}%")
print("smooth sweep (th=0.3):")
for w in [11, 15, 21, 31]:
    r, p = eval_threshold(eps, 0.3, w)
    print(f"  w={w} recall={r:.1f}% precision={p:.1f}%")
