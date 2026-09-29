import json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np

CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r4")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")
SW, A_DEN, MD, RR = 15, 1.0, 1500, 10
LAM = 0.8          # count-free penalty (model decides #cuts itself)

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

def dp_select(yc, score, N, min_dist=MD):
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

def refine(cuts, density, H, r=RR):
    out = []
    for c in cuts:
        lo = max(0, c - r); hi = min(H - 1, c + r)
        out.append(lo + int(np.argmin(density[lo:hi + 1])))
    return np.array(out, dtype=np.int32)

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
        gap = int(round(z["yc"][1] - z["yc"][0])) if len(z["yc"]) > 1 else 16
        eps.append({"series": e["series"], "episode": e["episode"], "yc": z["yc"], "score": score,
                    "seams": seams, "density": z["density"], "H": int(z["H"]), "gap": gap})
    print(f"ตอนทั้งหมดที่ประเมิน = {len(eps)} ตอน (held-out, ไม่เคยเห็นตอนเทรน)  จุดตัดจริง = "
          f"{sum(len(e['seams']) for e in eps)} เส้น\n")

    fixed = defaultdict(lambda: [0, 0])
    free = defaultdict(lambda: [0, 0, 0])  # matched, real, pred
    rows = []
    t0 = time.time()
    for ep in eps:
        N = len(ep["seams"])
        # --- MODE 1: fixed-N (บอกจำนวนเส้น) ---
        cuts = refine(dp_select(ep["yc"], ep["score"], N), ep["density"], ep["H"])
        m1 = match_1to1(cuts, ep["seams"])
        fixed[ep["series"]][0] += m1; fixed[ep["series"]][1] += N
        # --- MODE 2: count-free (AI ตัดสินใจจำนวนเอง = งานจริง) ---
        idxs = dp_free(ep["score"], 400, LAM, ep["gap"])
        cuts2 = refine(ep["yc"][idxs], ep["density"], ep["H"])
        K = len(cuts2)
        m2 = match_1to1(cuts2, ep["seams"])
        free[ep["series"]][0] += m2; free[ep["series"]][1] += N; free[ep["series"]][2] += K
        rows.append((ep["series"], ep["episode"], m1, N, m2, K))
    print("=" * 100)
    print("โหมด 1 — FIXED-N (±50px): บอกจำนวนเส้นให้ AI จำนวนเส้น = ของจริง")
    print("=" * 100)
    ft_m = sum(v[0] for v in fixed.values()); ft_n = sum(v[1] for v in fixed.values())
    for s in sorted(fixed):
        print(f"  {s:40s} {fixed[s][0]/fixed[s][1]*100:6.1f}%  ({fixed[s][0]}/{fixed[s][1]})")
    print(f"  TOTAL: {ft_m/ft_n*100:.1f}%  ({ft_m}/{ft_n})\n")

    print("=" * 100)
    print("โหมด 2 — AI ตัดสินใจจำนวนเส้นเอง (±50px) = งานจริงในแอป")
    print("  (K=เส้นที่ AI ตัดเอง, N=เส้นจริง, K/N<1 = ตัดน้อยไป, >1 = ตัดเกิน)")
    print("=" * 100)
    ff_m = sum(v[0] for v in free.values()); ff_n = sum(v[1] for v in free.values())
    ff_k = sum(v[2] for v in free.values())
    for s in sorted(free):
        m, n, k = free[s]
        rec = m / n * 100; prec = m / max(1, k) * 100
        f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0
        print(f"  {s:40s} recall={rec:5.1f}% prec={prec:5.1f}% F1={f1:5.1f}%  K/N={k/n:.2f}")
    rec = ff_m / ff_n * 100; prec = ff_m / max(1, ff_k) * 100
    f1 = 2 * rec * prec / (rec + prec)
    print(f"  TOTAL: recall={rec:.1f}% prec={prec:.1f}% F1={f1:.1f}%  (K/N={ff_k/ff_n:.2f})\n")

    print("=" * 100)
    print("ตัวอย่างตอน (เรียงตามความแม่น fixed-N): ดีสุด 3 / แย่สุด 3")
    print("=" * 100)
    rows_sorted = sorted(rows, key=lambda r: -(r[2] / r[3]))
    for label, group in (("ดีสุด", rows_sorted[:3]), ("แย่สุด", rows_sorted[-3:])):
        print(f"  --- {label} ---")
        for s, ep, m1, N, m2, K in group:
            print(f"    {s:30s} {ep:>8s} | fixed-N: {m1}/{N}={m1/N*100:5.1f}%  "
                  f"count-free: {m2}/{N} (AI ตัด {K} เส้น)")
    print(f"\nเวลารวม: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
