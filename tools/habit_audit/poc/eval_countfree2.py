import json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import eval_countfree as E

split = json.load(open(E.SPLIT, encoding="utf-8"))
eps = []
for e in split["val"]:
    name = E.safe(f"{e['series']}__{e['episode']}")
    z = np.load(E.CACHE / f"{name}.npz")
    seams = np.sort(z["seams"])
    if len(seams) == 0:
        continue
    sv = np.convolve(z["sv"], np.ones(E.SW) / E.SW, mode="same")
    score = sv - E.A_DEN * (z["density"][z["yc"]] / max(1.0, float(z["density"].max())))
    gap = int(z["yc"][1] - z["yc"][0])
    eps.append({"series": e["series"], "yc": z["yc"], "score": score, "seams": seams,
                "density": z["density"], "dmax": max(1.0, float(z["density"].max())),
                "H": int(z["H"]), "gap": gap})

def run(lam):
    tot_m = tot_n = tot_k = 0
    n_over = n_under = 0
    cnt_err = []
    for ep in eps:
        idxs = E.dp_free(ep["score"], E.MD, lam, ep["gap"])
        cuts = ep["yc"][idxs]
        den = ep["density"]
        out = []
        for c in cuts:
            lo = max(0, c - 10); hi = min(ep["H"] - 1, c + 10)
            out.append(lo + int(np.argmin(den[lo:hi + 1])))
        cuts = np.array(out, dtype=np.int32)
        K, N = len(cuts), len(ep["seams"])
        m = E.match_1to1(cuts, ep["seams"])
        tot_m += m; tot_n += N; tot_k += K
        if K > N: n_over += 1
        if K < N: n_under += 1
        cnt_err.append(K - N)
    cnt_err = np.array(cnt_err)
    rec = tot_m / tot_n * 100
    prec = tot_m / max(1, tot_k) * 100
    f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0
    return rec, prec, f1, tot_k / tot_n, n_over, n_under, cnt_err

print("fine lambda sweep (MD=400):")
print(f"{'lambda':>7} {'K/N':>6} {'over':>4} {'under':>5} {'recall':>7} {'prec':>6} {'F1':>5}")
rows = []
for lam in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
    rec, prec, f1, kn, n_o, n_u, ce = run(lam)
    rows.append((lam, kn, n_o, n_u, rec, prec, f1))
    print(f"{lam:7.2f} {kn:6.2f} {n_o:4d} {n_u:5d} {rec:7.1f} {prec:6.1f} {f1:5.1f}")

best = max(rows, key=lambda r: r[6])
print(f"\nbest F1 = {best[6]:.1f}% at lambda={best[0]}")

rec, prec, f1, kn, n_o, n_u, ce = run(best[0])
print(f"\ncount-error (K-N) distribution at lambda={best[0]}:")
for q in (10, 25, 50, 75, 90):
    print(f"  {q}% ของตอน: ต่างจากจริง {int(np.percentile(ce, q))} เส้น")
print(f"  ตอนที่ K==N พอดี: {(ce==0).sum()}/{len(ce)}")

# per-series
agg = defaultdict(lambda: [0, 0, 0])
for ep in eps:
    idxs = E.dp_free(ep["score"], E.MD, best[0], ep["gap"])
    cuts = ep["yc"][idxs]
    den = ep["density"]
    out = []
    for c in cuts:
        lo = max(0, c - 10); hi = min(ep["H"] - 1, c + 10)
        out.append(lo + int(np.argmin(den[lo:hi + 1])))
    cuts = np.array(out, dtype=np.int32)
    m = E.match_1to1(cuts, ep["seams"])
    agg[ep["series"]][0] += m
    agg[ep["series"]][1] += len(ep["seams"])
    agg[ep["series"]][2] += len(cuts)
print("\nper-series (F1-style, matched within +-50):")
for s in sorted(agg):
    m, n, k = agg[s]
    rec = m / n * 100; prec = m / k * 100
    f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0
    print(f"  {s:38s} recall={rec:5.1f}% prec={prec:5.1f}% F1={f1:5.1f}%  (K/N={k/n:.2f})")
