import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")

TOL = 50
MIN_DIST = 35
A_DEN = 0.75      # density prior weight (tuned on r3)
SMOOTH = 11
REFINE = 10
CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r4")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")

def safe(name):
    return (name.replace("\\", "_").replace("/", "_").replace(":", "_")
            .replace("*", "_").replace("?", "_").replace('"', "_")
            .replace("<", "_").replace(">", "_").replace("|", "_"))

def greedy_select(yc, score, N, min_dist=MIN_DIST):
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
    agg = defaultdict(lambda: {"m": 0, "r": 0})
    tot_m, tot_r = 0, 0
    n_ep = 0
    for e in split["val"]:
        name = safe(f"{e['series']}__{e['episode']}")
        p = CACHE / f"{name}.npz"
        if not p.exists():
            print("missing cache:", name)
            continue
        z = np.load(p)
        seams = np.sort(z["seams"])
        N = len(seams)
        if N == 0:
            continue
        n_ep += 1
        sv = np.convolve(z["sv"], np.ones(SMOOTH) / SMOOTH, mode="same")
        score = sv - A_DEN * (z["density"][z["yc"]] / max(1.0, float(z["density"].max())))
        cuts = greedy_select(z["yc"], score, N)
        # refine to local min density
        out = []
        H = int(z["H"]); den = z["density"]
        for c in cuts:
            lo = max(0, c - REFINE); hi = min(H - 1, c + REFINE)
            out.append(lo + int(np.argmin(den[lo:hi + 1])))
        cuts = np.array(out, dtype=np.int32)
        m = match_1to1(cuts, seams)
        agg[e["series"]]["m"] += m; agg[e["series"]]["r"] += N
        tot_m += m; tot_r += N
    print("=" * 70)
    print(f"FIXED-N +-{TOL}px | cnn_r4 | smooth={SMOOTH} a_den={A_DEN} dist={MIN_DIST} refine={REFINE}")
    print("=" * 70)
    for s in sorted(agg):
        print(f"  {s:38s} acc={agg[s]['m']/agg[s]['r']*100:5.1f}%  ({agg[s]['m']}/{agg[s]['r']})")
    print(f"  TOTAL ({n_ep} eps, {tot_r} cuts): {tot_m/tot_r*100:.1f}%")

if __name__ == "__main__":
    main()
