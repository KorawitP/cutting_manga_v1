import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import eval_dp2 as D

SW, A_DEN, MD, RR = 15, 1.0, 1500, 10
MODELS = {
    "r3": Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r3"),
    "r4": Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r4"),
}
split = json.load(open(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json", encoding="utf-8"))

def build(cache):
    out = []
    for e in split["val"]:
        name = D.safe(f"{e['series']}__{e['episode']}")
        z = np.load(cache / f"{name}.npz")
        seams = np.sort(z["seams"])
        if len(seams) == 0:
            continue
        out.append({"series": e["series"], "yc": z["yc"], "sv": z["sv"], "seams": seams,
                    "density": z["density"], "dmax": max(1.0, float(z["density"].max())),
                    "H": int(z["H"])})
    return out

eps3, eps4 = build(MODELS["r3"]), build(MODELS["r4"])
print("=" * 72)
print(f"DP SELECTION (smooth={SW} a_den={A_DEN} min_dist={MD} refine={RR}) | fixed-N +-50px")
print("=" * 72)
tot = {"r3": [0, 0], "r4": [0, 0]}
series = defaultdict(lambda: {"r3": [0, 0], "r4": [0, 0]})

for eps, key in ((eps3, "r3"), (eps4, "r4")):
    for ep in eps:
        sv = np.convolve(ep["sv"], np.ones(SW) / SW, mode="same")
        score = sv - A_DEN * (ep["density"][ep["yc"]] / ep["dmax"])
        cuts = D.dp_select(ep["yc"], score, len(ep["seams"]), MD)
        den = ep["density"]
        out = []
        for c in cuts:
            lo = max(0, c - RR); hi = min(ep["H"] - 1, c + RR)
            out.append(lo + int(np.argmin(den[lo:hi + 1])))
        cuts = np.array(out, dtype=np.int32)
        m = D.match_1to1(cuts, ep["seams"])
        tot[key][0] += m; tot[key][1] += len(ep["seams"])
        series[ep["series"]][key][0] += m; series[ep["series"]][key][1] += len(ep["seams"])

for s in sorted(series):
    a = series[s]
    r3 = a["r3"][0] / a["r3"][1] * 100
    r4 = a["r4"][0] / a["r4"][1] * 100
    print(f"  {s:38s} r3 {r3:5.1f}%  r4 {r4:5.1f}%  delta {r4-r3:+.1f}")
print("-" * 72)
print(f"  TOTAL | r3 {tot['r3'][0]/tot['r3'][1]*100:.1f}%  r4 {tot['r4'][0]/tot['r4'][1]*100:.1f}%  "
      f"delta {tot['r4'][0]/tot['r4'][1]*100 - tot['r3'][0]/tot['r3'][1]*100:+.1f}")
