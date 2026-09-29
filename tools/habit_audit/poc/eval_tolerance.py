import json, sys
from pathlib import Path
import numpy as np

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

def match_1to1(cuts, seams, tol):
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

print("FIXED-N accuracy vs tolerance (exactly same #cuts as real, 1:1 match)")
print(f"{'tol':>5} | {'r3':>6} {'r4':>6}  (r4-r3)")
for tol in (30, 50, 75, 100, 150, 200):
    row = []
    for key in ("r3", "r4"):
        m = sum(match_1to1(e[key], e["seams"], tol) for e in data.values())
        n = sum(len(e["seams"]) for e in data.values())
        row.append(m / n * 100)
    print(f"{tol:>5} | {row[0]:6.1f} {row[1]:6.1f}  ({row[1]-row[0]:+.1f})")
