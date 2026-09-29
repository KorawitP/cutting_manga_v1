import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

import sys
sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import eval_both as E

CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache")

def load_cache():
    eps = []
    for p in sorted(CACHE.glob("*.npz")):
        z = np.load(p)
        eps.append({"yc": z["yc"], "sv": z["sv"], "seams": z["seams"],
                    "seg": z["seg"], "density": z["density"], "H": int(z["H"])})
    return eps

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

eps = load_cache()
print(f"episodes={len(eps)}")
print("\n== threshold sweep (smooth=31) ==")
best = (0, None, None)
for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
    r, p = eval_threshold(eps, th, 31)
    print(f"  th={th:.2f} recall={r:.1f}% precision={p:.1f}%")
    if r > best[0]:
        best = (r, th, 31)
print("\n== smooth sweep (th=0.3) ==")
for w in [11, 21, 31, 41, 61]:
    r, p = eval_threshold(eps, 0.3, w)
    print(f"  w={w} recall={r:.1f}% precision={p:.1f}%")
    if r > best[0]:
        best = (r, 0.3, w)
print(f"\nBEST: th={best[1]} smooth={best[2]} recall={best[0]:.1f}%")
