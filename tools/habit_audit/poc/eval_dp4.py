import json, sys, time
from pathlib import Path
import numpy as np
import eval_dp2 as D

split = json.load(open(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json", encoding="utf-8"))
eps = []
for e in split["val"]:
    name = D.safe(f"{e['series']}__{e['episode']}")
    z = np.load(D.CACHE / f"{name}.npz")
    seams = np.sort(z["seams"])
    if len(seams) == 0:
        continue
    eps.append({"series": e["series"], "yc": z["yc"], "sv": z["sv"], "seams": seams,
                "density": z["density"], "dmax": max(1.0, float(z["density"].max())),
                "H": int(z["H"])})

# real gap distribution (from val seams + train features)
gaps = []
for e in eps:
    gaps.append(np.diff(e["seams"]))
for e in split["train"]:
    name = D.safe(f"{e['series']}__{e['episode']}")
    p = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\features") / f"{name}.npz"
    if p.exists():
        seams = np.sort(np.load(p)["seams"])
        if len(seams) > 1:
            gaps.append(np.diff(seams))
gaps = np.concatenate(gaps)
ps = np.percentile(gaps, [1, 5, 25, 50, 75, 95, 99])
print("real seam-gap percentiles:", [int(p) for p in ps])

def run(md, ad, sw, rr):
    tot_m = tot_r = 0
    for ep in eps:
        sv = np.convolve(ep["sv"], np.ones(sw) / sw, mode="same")
        score = sv - ad * (ep["density"][ep["yc"]] / ep["dmax"])
        cuts = D.dp_select(ep["yc"], score, len(ep["seams"]), md)
        if rr:
            den = ep["density"]
            out = []
            for c in cuts:
                lo = max(0, c - rr); hi = min(ep["H"] - 1, c + rr)
                out.append(lo + int(np.argmin(den[lo:hi + 1])))
            cuts = np.array(out, dtype=np.int32)
        tot_m += D.match_1to1(cuts, ep["seams"])
        tot_r += len(ep["seams"])
    return tot_m / tot_r * 100

t0 = time.time()
for md in (800, 1000, 1200, 1500, 1800):
    for ad in (1.0, 1.5):
        print(f"md={md:5d} a_den={ad:4.1f}  acc={run(md, ad, 15, 10):.1f}%")
print(f"done {time.time()-t0:.0f}s")
