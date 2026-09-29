import json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.signal import find_peaks

CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r4")
FEAT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\features")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")
SW, A_DEN, RR, MIN_GAP = 15, 1.0, 10, 700

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

def main():
    split = json.load(open(SPLIT, encoding="utf-8"))

    # ---- smooth logP(gap) from TRAIN gaps only ----
    train_gaps = []
    for e in split["train"]:
        name = safe(f"{e['series']}__{e['episode']}")
        p = FEAT / f"{name}.npz"
        if p.exists():
            seams = np.sort(np.load(p)["seams"])
            if len(seams) > 1:
                train_gaps.append(np.diff(seams))
    train_gaps = np.concatenate(train_gaps)
    edge0, edge1, step = 700, 16000, 200
    edges = np.arange(edge0, edge1 + step, step)
    hist, _ = np.histogram(train_gaps, bins=edges)
    density = hist / (len(train_gaps) * step)
    logd = np.log(np.maximum(density, 1e-8))
    centers = (edges[:-1] + edges[1:]) / 2
    def logp(gap):
        return np.interp(gap, centers, logd, left=logd[0] - 12, right=logd[-1] - 12)

    # ---- load val ----
    eps = []
    for e in split["val"]:
        name = safe(f"{e['series']}__{e['episode']}")
        z = np.load(CACHE / f"{name}.npz")
        seams = np.sort(z["seams"])
        if len(seams) == 0:
            continue
        sv = np.convolve(z["sv"], np.ones(SW) / SW, mode="same")
        score = sv - A_DEN * (z["density"][z["yc"]] / max(1.0, float(z["density"].max())))
        gap = int(round(z["yc"][1] - z["yc"][0])) if len(z["yc"]) > 1 else 16
        dist_s = max(1, int(MIN_GAP // gap))
        pk, _ = find_peaks(score, height=0.15, distance=dist_s)
        cand_y = z["yc"][pk]
        cand_w = score[pk]
        eps.append({"series": e["series"], "cand_y": cand_y, "cand_w": cand_w,
                    "seams": seams, "density": z["density"], "H": int(z["H"])})
    print(f"val eps={len(eps)} | avg candidates/ep = "
          f"{np.mean([len(x['cand_y']) for x in eps]):.0f}")

    def run(beta):
        tot_m = tot_r = 0
        for ep in eps:
            cand_y = ep["cand_y"]; cand_w = ep["cand_w"]
            N = len(ep["seams"])
            M = len(cand_y)
            if M < N:
                continue
            INF = -1e18
            dp_prev = cand_w.astype(np.float64).copy()
            dp_hist = [dp_prev.copy()]
            arg_hist = []
            for _ in range(2, N + 1):
                dp_cur = np.full(M, INF)
                arg = np.full(M, -1, dtype=np.int64)
                jj = np.arange(M)
                for i in range(M):
                    gaps = cand_y[i] - cand_y[:i]
                    if len(gaps) == 0:
                        continue
                    mask = gaps >= MIN_GAP
                    if not mask.any():
                        continue
                    vals = dp_prev[:i] + beta * logp(gaps)
                    vals[~mask] = INF
                    bj = int(np.argmax(vals))
                    if vals[bj] > INF * 0.5:
                        dp_cur[i] = cand_w[i] + vals[bj]
                        arg[i] = bj
                dp_hist.append(dp_cur)
                arg_hist.append(arg)
                dp_prev = dp_cur
            last = int(np.argmax(dp_prev))
            cuts = [int(cand_y[last])]
            for k in range(N - 1, 0, -1):
                last = int(arg_hist[k - 1][last])
                cuts.append(int(cand_y[last]))
            cuts = np.sort(np.array(cuts, dtype=np.int32))
            den = ep["density"]
            out = []
            for c in cuts:
                lo = max(0, c - RR); hi = min(ep["H"] - 1, c + RR)
                out.append(lo + int(np.argmin(den[lo:hi + 1])))
            cuts = np.array(out, dtype=np.int32)
            tot_m += match_1to1(cuts, ep["seams"])
            tot_r += N
        return tot_m / max(1, tot_r) * 100

    t0 = time.time()
    print("\nbeta sweep (beta = weight of segment-length prior):")
    res = {}
    for beta in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0):
        a = run(beta)
        res[beta] = a
        print(f"  beta={beta:4.1f} -> {a:.1f}%")
    best_beta = max(res, key=lambda k: res[k])
    print(f"\nbest beta={best_beta} -> {res[best_beta]:.1f}%  (beta=0 baseline = {res[0.0]:.1f}%)")

    agg = defaultdict(lambda: [0, 0])
    for ep in eps:
        cand_y = ep["cand_y"]; cand_w = ep["cand_w"]
        N = len(ep["seams"]); M = len(cand_y)
        if M < N:
            continue
        INF = -1e18
        dp_prev = cand_w.astype(np.float64).copy()
        dp_hist = [dp_prev.copy()]
        arg_hist = []
        for _ in range(2, N + 1):
            dp_cur = np.full(M, INF); arg = np.full(M, -1, dtype=np.int64)
            for i in range(M):
                gaps = cand_y[i] - cand_y[:i]
                if len(gaps) == 0:
                    continue
                mask = gaps >= MIN_GAP
                if not mask.any():
                    continue
                vals = dp_prev[:i] + best_beta * logp(gaps)
                vals[~mask] = INF
                bj = int(np.argmax(vals))
                if vals[bj] > INF * 0.5:
                    dp_cur[i] = cand_w[i] + vals[bj]; arg[i] = bj
            dp_hist.append(dp_cur); arg_hist.append(arg); dp_prev = dp_cur
        last = int(np.argmax(dp_prev))
        cuts = [int(cand_y[last])]
        for k in range(N - 1, 0, -1):
            last = int(arg_hist[k - 1][last]); cuts.append(int(cand_y[last]))
        cuts = np.sort(np.array(cuts, dtype=np.int32))
        den = ep["density"]
        out = []
        for c in cuts:
            lo = max(0, c - RR); hi = min(ep["H"] - 1, c + RR)
            out.append(lo + int(np.argmin(den[lo:hi + 1])))
        cuts = np.array(out, dtype=np.int32)
        agg[ep["series"]][0] += match_1to1(cuts, ep["seams"])
        agg[ep["series"]][1] += N
    print("\nper-series @ best beta:")
    for s in sorted(agg):
        print(f"  {s:38s} {agg[s][0]/agg[s][1]*100:5.1f}%  ({agg[s][0]}/{agg[s][1]})")
    print(f"\ntotal {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
