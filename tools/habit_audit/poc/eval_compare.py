import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

TOL = 50

def safe(name):
    return (name.replace("\\", "_").replace("/", "_").replace(":", "_")
            .replace("*", "_").replace("?", "_").replace('"', "_")
            .replace("<", "_").replace(">", "_").replace("|", "_"))

def greedy_select(yc, score, N, min_dist):
    order = np.argsort(score)[::-1]
    chosen = []
    for idx in order:
        y = int(yc[idx])
        ok = True
        for c in chosen:
            if abs(c - y) < min_dist:
                ok = False
                break
        if ok:
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

SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")
split = json.load(open(SPLIT, encoding="utf-8"))

MODELS = {
    "r3": (Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r3"), 11, 0.75, 35, 10),
    "r4": (Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r4"), 15, 1.00, 35, 20),
}

data = {}
for e in split["val"]:
    name = safe(f"{e['series']}__{e['episode']}")
    entry = {"series": e["series"], "seams": np.sort(np.load(
        MODELS["r3"][0] / f"{name}.npz")["seams"])}
    for key, (cache, sw, ad, md, rr) in MODELS.items():
        z = np.load(cache / f"{name}.npz")
        sv = np.convolve(z["sv"], np.ones(sw) / sw, mode="same")
        score = sv - ad * (z["density"][z["yc"]] / max(1.0, float(z["density"].max())))
        cuts = greedy_select(z["yc"], score, len(entry["seams"]), md)
        den = z["density"]
        out = []
        for c in cuts:
            lo = max(0, c - rr); hi = min(int(z["H"]) - 1, c + rr)
            out.append(lo + int(np.argmin(den[lo:hi + 1])))
        entry[key] = np.array(out, dtype=np.int32)
    data[name] = entry

agg = defaultdict(lambda: {"r3": [0, 0], "r4": [0, 0]})
for name, e in data.items():
    N = len(e["seams"])
    for key in ("r3", "r4"):
        m = match_1to1(e[key], e["seams"])
        agg[e["series"]][key][0] += m
        agg[e["series"]][key][1] += N

print("=" * 78)
print("FAIR COMPARE: each model at ITS OWN best config (fixed-N +-50px)")
print("r3 best = smooth11 a_den0.75 dist35 ref10 | r4 best = smooth15 a_den1.0 dist35 ref20")
print("=" * 78)
tot = {"r3": [0, 0], "r4": [0, 0]}
for s in sorted(agg):
    a = agg[s]
    for k in ("r3", "r4"):
        tot[k][0] += a[k][0]; tot[k][1] += a[k][1]
    print(f"  {s:38s} r3 {a['r3'][0]/a['r3'][1]*100:5.1f}%  r4 {a['r4'][0]/a['r4'][1]*100:5.1f}%   "
          f"delta {a['r4'][0]/a['r4'][1]*100 - a['r3'][0]/a['r3'][1]*100:+.1f}")
print("-" * 78)
print(f"  TOTAL (2856 cuts) | r3 {tot['r3'][0]/tot['r3'][1]*100:.1f}%  r4 {tot['r4'][0]/tot['r4'][1]*100:.1f}%  "
      f"delta {tot['r4'][0]/tot['r4'][1]*100 - tot['r3'][0]/tot['r3'][1]*100:+.1f}")
