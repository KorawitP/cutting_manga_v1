import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.signal import find_peaks

CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\effnet_cache")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")

def safe(name):
    return (name.replace("\\", "_").replace("/", "_").replace(":", "_")
            .replace("*", "_").replace("?", "_").replace('"', "_")
            .replace("<", "_").replace(">", "_").replace("|", "_"))

def match_1to1(cuts, seams, tol=50):
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

def dp_select(yc, score, N, min_dist):
    M = len(score)
    gap = int(round((yc[1] - yc[0]))) if M > 1 else 16
    idxgap = max(1, int(np.ceil(min_dist / gap)))
    INF = -1e18
    prev = score.astype(np.float64).copy()
    best_prev = []
    for _ in range(2, N + 1):
        pfm = np.maximum.accumulate(prev)
        run_val = np.full(M, INF)
        if M > idxgap:
            run_val[idxgap:] = pfm[:-idxgap]
        pa = np.full(M, -1, dtype=np.int64)
        bv, bi = INF, -1
        for t in range(M):
            v = prev[t]
            if v > bv:
                bv, bi = v, t
            pa[t] = bi
        bp = np.full(M, -1, dtype=np.int64)
        if M > idxgap:
            bp[idxgap:] = pa[:-idxgap]
        best_prev.append(bp)
        prev = run_val + score
    last = int(np.argmax(prev))
    cuts = [int(yc[last])]
    for bp in reversed(best_prev):
        last = int(bp[last])
        cuts.append(int(yc[last]))
    return np.sort(np.array(cuts, dtype=np.int32))

split = json.load(open(SPLIT, encoding="utf-8"))
eps = []
for e in split["val"]:
    name = safe(f"{e['series']}__{e['episode']}")
    z = np.load(CACHE / f"{name}.npz")
    seams = np.sort(z["seams"])
    if len(seams) == 0:
        continue
    eps.append({"series": e["series"], "yc": z["yc"], "sv": z["sv"], "seams": seams,
                "density": z["density"], "H": int(z["H"])})
print(f"eps={len(eps)} | original EfficientNet-B0 (app pipeline)")

print("\n--- (a) NATIVE app pipeline: find_peaks th=0.90, min_cut_distance=300 ---")
tot_m = tot_n = tot_k = 0
for ep in eps:
    pk, _ = find_peaks(ep["sv"], height=0.90, distance=max(1, 300 // 32))
    cuts = ep["yc"][pk]
    m = match_1to1(cuts, ep["seams"])
    tot_m += m; tot_n += len(ep["seams"]); tot_k += len(cuts)
print(f"  K/N = {tot_k/tot_n:.2f}  recall(matched/N)={tot_m/tot_n*100:.1f}%  "
      f"prec(matched/K)={tot_m/max(1,tot_k)*100:.1f}%  F1="
      f"{2*tot_m/(tot_n+max(1,tot_k))*100:.1f}%")

print("\n--- (a2) NATIVE with looser threshold th=0.5 ---")
tot_m = tot_n = tot_k = 0
for ep in eps:
    pk, _ = find_peaks(ep["sv"], height=0.5, distance=max(1, 300 // 32))
    cuts = ep["yc"][pk]
    m = match_1to1(cuts, ep["seams"])
    tot_m += m; tot_n += len(ep["seams"]); tot_k += len(cuts)
print(f"  K/N = {tot_k/tot_n:.2f}  recall={tot_m/tot_n*100:.1f}%  prec={tot_m/max(1,tot_k)*100:.1f}%  "
      f"F1={2*tot_m/(tot_n+max(1,tot_k))*100:.1f}%")

print("\n--- (b) FIXED-N DP (same metric as r4's 51.2%) ---")
print(f"{'tol':>5} | acc%")
for tol in (30, 50, 100, 150):
    for md in (1500,):
        tot_m = tot_r = 0
        for ep in eps:
            cuts = dp_select(ep["yc"], ep["sv"], len(ep["seams"]), md)
            tot_m += match_1to1(cuts, ep["seams"], tol)
            tot_r += len(ep["seams"])
        print(f"  +-{tol:4d} (md={md}) | {tot_m/tot_r*100:.1f}%")

print("\n--- (c) FIXED-N per-series @ +-50 (md=1500) ---")
agg = defaultdict(lambda: [0, 0])
for ep in eps:
    cuts = dp_select(ep["yc"], ep["sv"], len(ep["seams"]), 1500)
    agg[ep["series"]][0] += match_1to1(cuts, ep["seams"])
    agg[ep["series"]][1] += len(ep["seams"])
for s in sorted(agg):
    print(f"  {s:38s} {agg[s][0]/agg[s][1]*100:5.1f}%  ({agg[s][0]}/{agg[s][1]})")
