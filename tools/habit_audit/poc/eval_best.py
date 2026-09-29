import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import eval_dp2 as D

SW, A_DEN, MD, RR = 15, 1.0, 1500, 10
CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r4")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")
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

def select(ep):
    sv = np.convolve(ep["sv"], np.ones(SW) / SW, mode="same")
    score = sv - A_DEN * (ep["density"][ep["yc"]] / ep["dmax"])
    cuts = D.dp_select(ep["yc"], score, len(ep["seams"]), MD)
    den = ep["density"]
    out = []
    for c in cuts:
        lo = max(0, c - RR); hi = min(ep["H"] - 1, c + RR)
        out.append(lo + int(np.argmin(den[lo:hi + 1])))
    return np.array(out, dtype=np.int32)

# tolerance curve
print("=" * 70)
print("FINAL EVAL: cnn_r4 + DP selection | fixed-N (exactly same #cuts as real)")
print("=" * 70)
print("  tolerance |  accuracy")
for tol in (30, 50, 75, 100, 150, 200):
    tot_m = sum(D.match_1to1(select(ep), ep["seams"], tol) for ep in eps)
    tot_r = sum(len(ep["seams"]) for ep in eps)
    print(f"  +-{tol:4d}px  |  {tot_m/tot_r*100:.1f}%  ({tot_m}/{tot_r})")

print("\n  per-series @ +-50px:")
agg = defaultdict(lambda: [0, 0])
for ep in eps:
    m = D.match_1to1(select(ep), ep["seams"])
    agg[ep["series"]][0] += m
    agg[ep["series"]][1] += len(ep["seams"])
for s in sorted(agg):
    print(f"    {s:38s} {agg[s][0]/agg[s][1]*100:5.1f}%  ({agg[s][0]}/{agg[s][1]})")
