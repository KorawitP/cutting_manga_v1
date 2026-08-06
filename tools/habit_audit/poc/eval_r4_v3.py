import json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np

POC = Path(__file__).resolve().parent
CACHE = POC / "cnn_cache_r4_v3"
SPLIT = POC / "split_v3.json"
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

def dp_free(score, min_dist, lam, gap):
    M = len(score)
    idxgap = max(1, int(np.ceil(min_dist / gap)))
    INF = -1e18
    dp = np.full(M, INF)
    bp = np.full(M, -1, dtype=np.int64)
    best_val, best_idx = INF, -1
    for i in range(M):
        j = i - idxgap
        if j >= 0 and dp[j] > best_val:
            best_val, best_idx = dp[j], j
        prev = best_val if best_val > 0 else 0.0
        dp[i] = score[i] - lam + prev
        bp[i] = best_idx if best_val > 0 else -1
    last = int(np.argmax(dp))
    idxs = [last]
    while bp[last] != -1:
        last = int(bp[last])
        idxs.append(last)
    return np.sort(np.array(idxs, dtype=np.int64))

def refine(cuts, density, H, r=10):
    out = []
    for c in cuts:
        lo = max(0, c - r); hi = min(H - 1, c + r)
        out.append(lo + int(np.argmin(density[lo:hi + 1])))
    return np.array(out, dtype=np.int32)

def main():
    split = json.load(open(SPLIT, encoding="utf-8"))
    eps = []
    n_skip = 0
    for e in split["val"]:
        name = safe(f"{e['series']}__{e['episode']}")
        p = CACHE / f"{name}.npz"
        if not p.exists():
            n_skip += 1
            continue
        z = np.load(p)
        seams = np.sort(z["seams"])
        if len(seams) == 0:
            continue
        eps.append({"series": e["series"], "episode": e["episode"], "yc": z["yc"], "sv": z["sv"],
                    "seams": seams, "density": z["density"], "dmax": max(1.0, float(z["density"].max())),
                    "H": int(z["H"])})
    print(f"episodes={len(eps)} (skipped cache missing={n_skip})  real_seams={sum(len(e['seams']) for e in eps)}")

    # ---- DP sweep (fixed-N) ----
    t0 = time.time()
    results = {}
    best = (0, None)
    for sw in (11, 15, 19):
        for ad in (0.75, 1.0, 1.5):
            for md in (1000, 1500, 2000):
                tot_m = tot_r = 0
                for ep in eps:
                    sv = np.convolve(ep["sv"], np.ones(sw) / sw, mode="same")
                    score = sv - ad * (ep["density"][ep["yc"]] / ep["dmax"])
                    cuts = refine(dp_select(ep["yc"], score, len(ep["seams"]), md), ep["density"], ep["H"])
                    tot_m += match_1to1(cuts, ep["seams"])
                    tot_r += len(ep["seams"])
                acc = tot_m / tot_r * 100
                results[(sw, ad, md)] = acc
                if acc > best[0]:
                    best = (acc, (sw, ad, md))
    print(f"sweep done {time.time()-t0:.0f}s")
    for cfg, acc in sorted(results.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  smooth={cfg[0]:3d} a_den={cfg[1]:4.2f} min_dist={cfg[2]:5d}  acc={acc:.1f}%")
    sw, ad, md = best[1]
    print(f"BEST fixed-N @+-50: {best[0]:.1f}%  cfg={best[1]}")

    # ---- tolerance curve at best ----
    print("\n  tolerance | accuracy (fixed-N, best cfg)")
    for tol in (30, 50, 75, 100, 150, 200):
        tot_m = tot_r = 0
        for ep in eps:
            sv = np.convolve(ep["sv"], np.ones(sw) / sw, mode="same")
            score = sv - ad * (ep["density"][ep["yc"]] / ep["dmax"])
            cuts = refine(dp_select(ep["yc"], score, len(ep["seams"]), md), ep["density"], ep["H"])
            tot_m += match_1to1(cuts, ep["seams"], tol)
            tot_r += len(ep["seams"])
        print(f"  +-{tol:4d}px  |  {tot_m/tot_r*100:.1f}%  ({tot_m}/{tot_r})")

    # ---- per-series fixed-N ----
    print("\n  per-series @ +-50 (fixed-N):")
    agg = defaultdict(lambda: [0, 0])
    for ep in eps:
        sv = np.convolve(ep["sv"], np.ones(sw) / sw, mode="same")
        score = sv - ad * (ep["density"][ep["yc"]] / ep["dmax"])
        cuts = refine(dp_select(ep["yc"], score, len(ep["seams"]), md), ep["density"], ep["H"])
        agg[ep["series"]][0] += match_1to1(cuts, ep["seams"])
        agg[ep["series"]][1] += len(ep["seams"])
    for s in sorted(agg):
        print(f"    {s[:40]:40s} {agg[s][0]/agg[s][1]*100:5.1f}%  ({agg[s][0]}/{agg[s][1]})")

    # ---- count-free ----
    print("\n  count-free (AI decides #cuts) sweep over lambda:")
    bestf = (0, None)
    for lam in (0.5, 0.8, 1.0, 1.2):
        tot_m = tot_r = tot_k = 0
        for ep in eps:
            sv = np.convolve(ep["sv"], np.ones(sw) / sw, mode="same")
            score = sv - ad * (ep["density"][ep["yc"]] / ep["dmax"])
            gap = int(round(ep["yc"][1] - ep["yc"][0])) if len(ep["yc"]) > 1 else 16
            idxs = dp_free(score, 400, lam, gap)
            cuts = refine(ep["yc"][idxs], ep["density"], ep["H"])
            m = match_1to1(cuts, ep["seams"])
            tot_m += m; tot_r += len(ep["seams"]); tot_k += len(cuts)
        rec = tot_m / tot_r
        prec = tot_m / max(1, tot_k)
        f1 = 2 * rec * prec / (rec + prec) if rec + prec > 0 else 0
        print(f"    lam={lam:.2f}  recall={rec*100:.1f}%  prec={prec*100:.1f}%  F1={f1*100:.1f}%  (K/N={tot_k/tot_r:.2f})")
        if f1 > bestf[0]:
            bestf = (f1, lam)
    print(f"BEST count-free F1: {bestf[0]*100:.1f}%  lam={bestf[1]}")

    # ---- per-series count-free at best lam ----
    lam = bestf[1]
    print(f"\n  per-series count-free (lam={lam}):")
    agg = defaultdict(lambda: [0, 0, 0])
    for ep in eps:
        sv = np.convolve(ep["sv"], np.ones(sw) / sw, mode="same")
        score = sv - ad * (ep["density"][ep["yc"]] / ep["dmax"])
        gap = int(round(ep["yc"][1] - ep["yc"][0])) if len(ep["yc"]) > 1 else 16
        idxs = dp_free(score, 400, lam, gap)
        cuts = refine(ep["yc"][idxs], ep["density"], ep["H"])
        m = match_1to1(cuts, ep["seams"])
        agg[ep["series"]][0] += m; agg[ep["series"]][1] += len(ep["seams"]); agg[ep["series"]][2] += len(cuts)
    for s in sorted(agg):
        m, r, k = agg[s]
        rec = m / r; prec = m / max(1, k)
        f1 = 2 * rec * prec / (rec + prec) if rec + prec > 0 else 0
        print(f"    {s[:40]:40s} F1={f1*100:5.1f}%  (r={m}/{r} k={k})")

if __name__ == "__main__":
    main()

