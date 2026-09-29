import json, sys, time
from pathlib import Path
import numpy as np
import eval_dp2 as D

CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r4")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")
SW, RR = 15, 10

split = json.load(open(SPLIT, encoding="utf-8"))
eps = []
for e in split["val"]:
    name = D.safe(f"{e['series']}__{e['episode']}")
    z = np.load(CACHE / f"{name}.npz")
    seams = np.sort(z["seams"])
    if len(seams) == 0:
        continue
    eps.append({"series": e["series"], "yc": z["yc"], "sv": z["sv"], "seams": seams,
                "density": z["density"], "dmax": max(1.0, float(z["density"].max())),
                "H": int(z["H"])})

def run(ad, md):
    tot_m = tot_r = 0
    for ep in eps:
        sv = np.convolve(ep["sv"], np.ones(SW) / SW, mode="same")
        score = sv - ad * (ep["density"][ep["yc"]] / ep["dmax"])
        cuts = D.dp_select(ep["yc"], score, len(ep["seams"]), md)
        den = ep["density"]
        out = []
        for c in cuts:
            lo = max(0, c - RR); hi = min(ep["H"] - 1, c + RR)
            out.append(lo + int(np.argmin(den[lo:hi + 1])))
        cuts = np.array(out, dtype=np.int32)
        tot_m += D.match_1to1(cuts, ep["seams"])
        tot_r += len(ep["seams"])
    return tot_m / tot_r * 100

t0 = time.time()
print("density-prior strength x min_dist (full-sample DP):")
print(f"{'a_den':>5} | " + " | ".join(f"md={m}" for m in (1200, 1500, 1800)))
for ad in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
    row = []
    for md in (1200, 1500, 1800):
        row.append(f"{run(ad, md):5.1f}")
    print(f"{ad:5.1f} | " + " | ".join(f"{v}" for v in row))
print(f"done {time.time()-t0:.0f}s")
