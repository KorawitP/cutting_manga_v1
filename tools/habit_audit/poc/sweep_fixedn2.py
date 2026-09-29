import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import eval_both as E

TOL = 50
CACHE_R1 = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache")
CACHE_R3 = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r3")
FEAT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\features")
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
    for e in val:
        name = safe(f"{e['series']}__{e['episode']}")
        z1 = np.load(CACHE_R1 / f"{name}.npz")
        z3 = np.load(CACHE_R3 / f"{name}.npz")
        seams = np.sort(z3["seams"])
        if len(seams) == 0:
            continue
        fz = np.load(FEAT / f"{name}.npz")
        gw = fz["feats"][:, 4]
        eps.append({
            "seams": seams, "yc": z3["yc"], "sv1": z1["sv"], "sv3": z3["sv"],
            "align": np.array_equal(z1["yc"], z3["yc"]),
            "density": z3["density"], "dmax": max(1.0, float(z3["density"].max())),
            "gw": gw, "gwmax": max(1.0, float(gw.max())), "H": int(z3["H"]),
        })
    print(f"loaded {len(eps)} episodes (fixed-N, tol=+-{TOL}px)")

    def smooth(sv, w):
        return np.convolve(sv, np.ones(w) / w, mode="same")

    models_cfg = {"cnn_avg": (9, 11, 13), "cnn_r3": (9, 11, 13), "cnn_r1": (11,)}
    configs = []
    for m, sws in models_cfg.items():
        for sw in sws:
            for ad in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
                for agw in (0.0, 0.1, 0.2, 0.3, 0.5):
                    for md in (25, 30, 35):
                        configs.append((m, sw, ad, agw, md))
    print(f"configs: {len(configs)}")

    t0 = time.time()
    results = {}
    for cfg in configs:
        m, sw, ad, agw, md = cfg
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
            if ad:
                score = score - ad * (ep["density"][yc] / ep["dmax"])
            if agw:
                score = score + agw * (ep["gw"][yc] / ep["gwmax"])
            cuts = greedy_select(yc, score, N, md)
            tot_m += match_1to1(cuts, seams)
            tot_r += N
        results[cfg] = tot_m / max(1, tot_r) * 100
    print(f"sweep done in {time.time()-t0:.0f}s")

    best = sorted(results.items(), key=lambda kv: -kv[1])
    print("\n==== TOP 30 CONFIGS ====")
    print(f"{'model':9s} {'smooth':>6s} {'a_den':>6s} {'a_gw':>6s} {'dist':>4s}  acc%")
    for cfg, acc in best[:30]:
        m, sw, ad, agw, md = cfg
        print(f"{m:9s} {sw:6d} {ad:6.2f} {agw:6.2f} {md:4d}  {acc:5.1f}")

if __name__ == "__main__":
    main()
