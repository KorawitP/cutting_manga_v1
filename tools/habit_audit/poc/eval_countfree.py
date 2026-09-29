import json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np

CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r4")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")
SW, A_DEN, MD = 15, 1.0, 400

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

def dp_free(score, min_dist, lam, gap):
    """Model decides #cuts: maximize sum(score) - lam*count, cuts >= min_dist apart."""
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

def main():
    split = json.load(open(SPLIT, encoding="utf-8"))
    eps = []
    for e in split["val"]:
        name = safe(f"{e['series']}__{e['episode']}")
        z = np.load(CACHE / f"{name}.npz")
        seams = np.sort(z["seams"])
        if len(seams) == 0:
            continue
        sv = np.convolve(z["sv"], np.ones(SW) / SW, mode="same")
        score = sv - A_DEN * (z["density"][z["yc"]] / max(1.0, float(z["density"].max())))
        gap = int(z["yc"][1] - z["yc"][0])
        eps.append({"series": e["series"], "yc": z["yc"], "score": score, "seams": seams,
                    "density": z["density"], "dmax": max(1.0, float(z["density"].max())),
                    "H": int(z["H"]), "gap": gap})
    print(f"eps={len(eps)} | model decides #cuts (penalized DP)")

    t0 = time.time()
    print(f"\n{'lambda':>7} {'K~N?':>7} {'over':>5} {'under':>5} {'recall':>7} {'prec':>6} {'F1':>5}  matched")
    rows = []
    for lam in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2):
        tot_m = tot_n = tot_k = 0
        n_over = n_under = 0
        for ep in eps:
            idxs = dp_free(ep["score"], MD, lam, ep["gap"])
            cuts = ep["yc"][idxs]
            den = ep["density"]
            out = []
            for c in cuts:
                lo = max(0, c - 10); hi = min(ep["H"] - 1, c + 10)
                out.append(lo + int(np.argmin(den[lo:hi + 1])))
            cuts = np.array(out, dtype=np.int32)
            K, N = len(cuts), len(ep["seams"])
            m = match_1to1(cuts, ep["seams"])
            tot_m += m; tot_n += N; tot_k += K
            if K > N: n_over += 1
            if K < N: n_under += 1
        rec = tot_m / tot_n * 100
        prec = tot_m / max(1, tot_k) * 100
        f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0
        rows.append((lam, tot_k / tot_n, n_over, n_under, rec, prec, f1, tot_m))
        print(f"{lam:7.1f} {tot_k/tot_n:7.2f} {n_over:5d} {n_under:5d} "
              f"{rec:7.1f} {prec:6.1f} {f1:5.1f}  {tot_m}")
    print(f"done {time.time()-t0:.0f}s")
    print("\nหมายเหตุ: K=เส้นที่ AI ตัดเอง, N=เส้นจริง, over/under=จำนวนตอนที่ตัดเกิน/ขาด")

if __name__ == "__main__":
    main()
