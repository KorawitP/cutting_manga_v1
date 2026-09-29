import json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np

CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r4")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")
TOL = 50

def safe(name):
    return (name.replace("\\", "_").replace("/", "_").replace(":", "_")
            .replace("*", "_").replace("?", "_").replace('"', "_")
            .replace("<", "_").replace(">", "_").replace("|", "_"))

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

def dp_select(yc, score, N, min_dist):
    """Exactly N cuts maximizing sum(score), min_dist px apart (O(N*M))."""
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
        # prefix argmax of prev (O(M) python loop)
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

def main():
    split = json.load(open(SPLIT, encoding="utf-8"))
    eps = []
    for e in split["val"]:
        name = safe(f"{e['series']}__{e['episode']}")
        z = np.load(CACHE / f"{name}.npz")
        seams = np.sort(z["seams"])
        if len(seams) == 0:
            continue
        eps.append({"series": e["series"], "yc": z["yc"], "sv": z["sv"],
                    "seams": seams, "density": z["density"], "dmax": max(1.0, float(z["density"].max())),
                    "H": int(z["H"])})
    print(f"episodes={len(eps)}")

    configs = []
    for sw in (11, 15):
        for ad in (0.75, 1.0, 1.5):
            for md in (100, 150, 200, 300, 400):
                for rr in (10,):
                    configs.append((sw, ad, md, rr))
    print(f"configs={len(configs)}")
    t0 = time.time()
    results = {}
    best = (0, None)
    for cfg in configs:
        sw, ad, md, rr = cfg
        tot_m = tot_r = 0
        for ep in eps:
            sv = np.convolve(ep["sv"], np.ones(sw) / sw, mode="same")
            score = sv - ad * (ep["density"][ep["yc"]] / ep["dmax"])
            cuts = dp_select(ep["yc"], score, len(ep["seams"]), md)
            if rr:
                den = ep["density"]
                out = []
                for c in cuts:
                    lo = max(0, c - rr); hi = min(ep["H"] - 1, c + rr)
                    out.append(lo + int(np.argmin(den[lo:hi + 1])))
                cuts = np.array(out, dtype=np.int32)
            tot_m += match_1to1(cuts, ep["seams"])
            tot_r += len(ep["seams"])
        acc = tot_m / tot_r * 100
        results[cfg] = acc
        if acc > best[0]:
            best = (acc, cfg)
    print(f"sweep done {time.time()-t0:.0f}s")
    for cfg, acc in sorted(results.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  smooth={cfg[0]:3d} a_den={cfg[1]:4.2f} min_dist={cfg[2]:4d} refine={cfg[3]:3d}  acc={acc:.1f}%")
    print(f"\nBEST: acc={best[0]:.1f}% cfg={best[1]}")

    # per-series at best
    sw, ad, md, rr = best[1]
    agg = defaultdict(lambda: [0, 0])
    for ep in eps:
        sv = np.convolve(ep["sv"], np.ones(sw) / sw, mode="same")
        score = sv - ad * (ep["density"][ep["yc"]] / ep["dmax"])
        cuts = dp_select(ep["yc"], score, len(ep["seams"]), md)
        if rr:
            den = ep["density"]
            out = []
            for c in cuts:
                lo = max(0, c - rr); hi = min(ep["H"] - 1, c + rr)
                out.append(lo + int(np.argmin(den[lo:hi + 1])))
            cuts = np.array(out, dtype=np.int32)
        agg[ep["series"]][0] += match_1to1(cuts, ep["seams"])
        agg[ep["series"]][1] += len(ep["seams"])
    for s in sorted(agg):
        print(f"  {s:38s} {agg[s][0]/agg[s][1]*100:5.1f}%  ({agg[s][0]}/{agg[s][1]})")

if __name__ == "__main__":
    main()
