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

def main():
    split = json.load(open(SPLIT, encoding="utf-8"))
    eps = []
    for e in split["val"]:
        name = safe(f"{e['series']}__{e['episode']}")
        z = np.load(CACHE / f"{name}.npz")
        seams = np.sort(z["seams"])
        if len(seams) == 0:
            continue
        eps.append({"series": e["series"], "seams": seams, "yc": z["yc"], "sv": z["sv"],
                    "density": z["density"], "dmax": max(1.0, float(z["density"].max())),
                    "H": int(z["H"])})
    print(f"loaded {len(eps)} episodes")

    def smooth(sv, w):
        return np.convolve(sv, np.ones(w) / w, mode="same")

    configs = []
    for sw in (7, 9, 11, 13, 15):
        for ad in (0.0, 0.25, 0.5, 0.75, 1.0):
            for md in (35, 40):
                for rr in (0, 10, 20):
                    configs.append((sw, ad, md, rr))
    print(f"configs: {len(configs)}")

    t0 = time.time()
    results = {}
    best_cfg = None
    best_acc = 0
    best_per_series = None
    for cfg in configs:
        sw, ad, md, rr = cfg
        tot_m = tot_r = 0
        ps = defaultdict(lambda: [0, 0])
        for ep in eps:
            seams, N = ep["seams"], len(ep["seams"])
            score = smooth(ep["sv"], sw)
            if ad:
                score = score - ad * (ep["density"][ep["yc"]] / ep["dmax"])
            cuts = greedy_select(ep["yc"], score, N, md)
            if rr:
                den = ep["density"]
                out = []
                for c in cuts:
                    lo = max(0, c - rr); hi = min(ep["H"] - 1, c + rr)
                    out.append(lo + int(np.argmin(den[lo:hi + 1])))
                cuts = np.array(out, dtype=np.int32)
            m = match_1to1(cuts, seams)
            tot_m += m; tot_r += N
            ps[ep["series"]][0] += m; ps[ep["series"]][1] += N
        acc = tot_m / max(1, tot_r) * 100
        results[cfg] = acc
        if acc > best_acc:
            best_acc = acc; best_cfg = cfg; best_per_series = ps
    print(f"sweep done {time.time()-t0:.0f}s")

    best = sorted(results.items(), key=lambda kv: -kv[1])
    print("\nTOP 20:")
    print(f"{'smooth':>6} {'a_den':>6} {'dist':>4} {'refine':>6}  acc%")
    for cfg, acc in best[:20]:
        sw, ad, md, rr = cfg
        print(f"{sw:6d} {ad:6.2f} {md:4d} {rr:6d}  {acc:5.1f}")

    print("\nBEST per-series (cfg=%s):" % (best_cfg,))
    for s in sorted(best_per_series):
        m, r = best_per_series[s]
        print(f"  {s:38s} {m/max(1,r)*100:5.1f}%  ({m}/{r})")

if __name__ == "__main__":
    main()
