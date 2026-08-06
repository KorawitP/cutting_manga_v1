import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import eval_both as E

TOL = 50
CACHE_R1 = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache")
CACHE_R3 = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r3")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")

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

def refine_to_density(cuts, density, r):
    out = []
    H = len(density)
    for c in cuts:
        lo = max(0, c - r); hi = min(H - 1, c + r)
        out.append(lo + int(np.argmin(density[lo:hi + 1])))
    return np.array(out, dtype=np.int32)

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
    val = split["val"]
    eps = []
    gaps = []
    for e in val:
        name = safe(f"{e['series']}__{e['episode']}")
        z1 = np.load(CACHE_R1 / f"{name}.npz")
        z3 = np.load(CACHE_R3 / f"{name}.npz")
        seams = np.sort(z3["seams"])
        if len(seams) == 0:
            continue
        gaps.append(np.diff(seams))
        eps.append({
            "name": name, "seams": seams, "seg": z3["seg"], "yc": z3["yc"],
            "sv1": z1["sv"], "sv3": z3["sv"], "align": np.array_equal(z1["yc"], z3["yc"]),
            "density": z3["density"], "dmax": max(1.0, float(z3["density"].max())),
            "H": int(z3["H"]),
        })
    gaps = np.concatenate(gaps)
    print(f"loaded {len(eps)} episodes | seam min-gap: p5={np.percentile(gaps,5):.0f} "
          f"min={gaps.min()} median={np.median(gaps):.0f}")

    def smooth(sv, w):
        return np.convolve(sv, np.ones(w) / w, mode="same")

    models_cfg = {"cnn_r3": (7, 8, 9, 10, 11), "cnn_avg": (7, 8, 9, 10, 11), "cnn_r1": (9,)}
    configs = []
    for m, sws in models_cfg.items():
        for sw in sws:
            for ad in (0.3, 0.4, 0.5, 0.6, 0.75):
                for md in (35, 40, 45):
                    for rr in (0, 5, 10):
                        configs.append((m, sw, ad, md, rr))
    print(f"configs: {len(configs)}")

    t0 = time.time()
    results = {}
    for cfg in configs:
        m, sw, ad, md, rr = cfg
        tot_m, tot_r = 0, 0
        for ep in eps:
            seams, N = ep["seams"], len(ep["seams"])
            if m == "cnn_r3":
                yc, score = ep["yc"], smooth(ep["sv3"], sw)
            elif m == "cnn_r1":
                yc, score = ep["yc"], smooth(ep["sv1"], sw)
            else:
                if not ep["align"]:
                    continue
                yc, score = ep["yc"], smooth((ep["sv3"] + ep["sv1"]) * 0.5, sw)
            score = score - ad * (ep["density"][yc] / ep["dmax"])
            cuts = greedy_select(yc, score, N, md)
            if rr:
                cuts = refine_to_density(cuts, ep["density"], rr)
            tot_m += match_1to1(cuts, seams)
            tot_r += N
        results[cfg] = tot_m / max(1, tot_r) * 100
    print(f"sweep done in {time.time()-t0:.0f}s")

    best = sorted(results.items(), key=lambda kv: -kv[1])
    print("\n==== TOP 30 CONFIGS ====")
    print(f"{'model':9s} {'smooth':>6s} {'a_den':>6s} {'dist':>4s} {'refine':>6s}  acc%")
    for cfg, acc in best[:30]:
        m, sw, ad, md, rr = cfg
        print(f"{m:9s} {sw:6d} {ad:6.2f} {md:4d} {rr:6d}  {acc:5.1f}")

if __name__ == "__main__":
    main()
