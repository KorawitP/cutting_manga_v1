import json, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import eval_both as E
import joblib

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
    gbm = joblib.load(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\gbm_r1.joblib")

    eps = []
    for e in val:
        name = safe(f"{e['series']}__{e['episode']}")
        z1 = np.load(CACHE_R1 / f"{name}.npz")
        z3 = np.load(CACHE_R3 / f"{name}.npz")
        seams = np.sort(z3["seams"])
        if len(seams) == 0:
            continue
        density = z3["density"]
        H = int(z3["H"])
        # CNN scores (window centers)
        yc = z3["yc"]
        align = np.array_equal(z1["yc"], yc)
        sv1, sv3 = z1["sv"], z3["sv"]
        # GBM score (pixel rows)
        fz = np.load(FEAT / f"{name}.npz")
        feats = E.norm_feats(fz["feats"])
        probs = gbm.predict_proba(feats)[:, 1]
        gsv = probs.copy()
        gsv[feats[:, 4] < 5] *= 0.3
        eps.append({
            "name": name, "seams": seams, "density": density, "H": H,
            "yc": yc, "sv1": sv1, "sv3": sv3, "align": align,
            "gsv": gsv, "rows": np.arange(H, dtype=np.int32),
            "dmax": max(1.0, float(density.max())),
        })
    print(f"loaded {len(eps)} episodes (fixed-N eval, tol=+-{TOL}px)")

    def smooth(sv, w):
        return np.convolve(sv, np.ones(w) / w, mode="same")

    def prior(score, dnorm, alpha):
        return score - alpha * dnorm if alpha else score

    configs = []  # (model, smooth_w, alpha, min_dist, refine_r, peak_first)
    models_cfg = {
        "cnn_r3": (7, 11, 15, 21), "cnn_r1": (7, 11, 15),
        "cnn_avg": (7, 11, 15), "gbm": (21, 31, 41), "hybrid": (11, 21),
    }
    for m, sws in models_cfg.items():
        for sw in sws:
            for alpha in (0.0, 0.05, 0.1, 0.2):
                for md in (30,):
                    for rr in (0, 20, 30):
                        for pf in (False, True):
                            configs.append((m, sw, alpha, md, rr, pf))
    print(f"configs: {len(configs)}")

    t0 = time.time()
    results = {}
    for cfg in configs:
        m, sw, alpha, md, rr, pf = cfg
        tot_m, tot_r = 0, 0
        for ep in eps:
            seams, N = ep["seams"], len(ep["seams"])
            if m == "cnn_r3":
                yc, score = ep["yc"], smooth(ep["sv3"], sw)
            elif m == "cnn_r1":
                yc, score = ep["yc"], smooth(ep["sv1"], sw)
            elif m == "cnn_avg":
                if not ep["align"]:
                    continue
                yc, score = ep["yc"], smooth((ep["sv3"] + ep["sv1"]) * 0.5, sw)
            elif m == "gbm":
                yc, score = ep["rows"], smooth(ep["gsv"], sw)
            elif m == "hybrid":
                # upsample CNN score to pixel rows, average with GBM
                cnn_px = np.interp(ep["rows"], ep["yc"], smooth(ep["sv3"], sw))
                score = (cnn_px + smooth(ep["gsv"], sw)) * 0.5
                yc = ep["rows"]
            if alpha:
                dnorm = ep["density"][yc] / ep["dmax"]
                score = prior(score, dnorm, alpha)
            cuts = greedy_select(yc, score, N, md)
            if rr:
                cuts = refine_to_density(cuts, ep["density"], rr)
            tot_m += match_1to1(cuts, seams)
            tot_r += N
        results[cfg] = tot_m / max(1, tot_r) * 100
    print(f"sweep done in {time.time()-t0:.0f}s")

    best = sorted(results.items(), key=lambda kv: -kv[1])
    print("\n==== TOP 25 CONFIGS ====")
    print(f"{'model':9s} {'smooth':>6s} {'alpha':>6s} {'dist':>4s} {'refine':>6s} {'peak1st':>7s}  acc%")
    for cfg, acc in best[:25]:
        m, sw, alpha, md, rr, pf = cfg
        print(f"{m:9s} {sw:6d} {alpha:6.2f} {md:4d} {rr:6d} {str(pf):>7s}  {acc:5.1f}")

if __name__ == "__main__":
    main()
