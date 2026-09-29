import json, sys
from pathlib import Path
import numpy as np

CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r4")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")
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
    """Exactly N cuts maximizing sum(score) with min_dist spacing (DP)."""
    M = len(score)
    idxgap = max(1, int(np.ceil(min_dist / 16)))
    INF = -1e18
    prev = score.copy()  # dp for 1 cut
    best_prev = []
    for _ in range(2, N + 1):
        pfm = np.maximum.accumulate(prev)
        run = np.full(M, INF)
        if M > idxgap:
            run[idxgap:] = pfm[:-idxgap]
        cur = score + run
        # best_prev position for backtracking: argmax over j<=i-idxgap of prev[j]
        bp = np.full(M, -1)
        if M > idxgap:
            for i in range(idxgap, M):
                j0 = i - idxgap
                bp[i] = int(np.argmax(prev[:j0 + 1]))
        best_prev.append(bp)
        prev = cur
    last = int(np.argmax(prev))
    cuts = [int(yc[last])]
    for bp in reversed(best_prev):
        last = int(bp[last])
        cuts.append(int(yc[last]))
    return np.sort(np.array(cuts, dtype=np.int32))

def main():
    split = json.load(open(SPLIT, encoding="utf-8"))
    eps = []
    for e in split["val"]:
        name = safe(f"{e['series']}__{e['episode']}")
        z = np.load(CACHE / f"{name}.npz")
        seams = np.sort(z["seams"])
        if len(seams) == 0:
            continue
        sv = np.convolve(z["sv"], np.ones(15) / 15, mode="same")
        score = sv - 1.0 * (z["density"][z["yc"]] / max(1.0, float(z["density"].max())))
        eps.append({"series": e["series"], "yc": z["yc"], "score": score,
                    "seams": seams, "density": z["density"], "H": int(z["H"])})
    print(f"episodes={len(eps)}  DP selection (min_dist=35)")

    for md in (35, 60, 100):
        tot_m = tot_r = 0
        for ep in eps:
            cuts = dp_select(ep["yc"], ep["score"], len(ep["seams"]), md)
            tot_m += match_1to1(cuts, ep["seams"])
            tot_r += len(ep["seams"])
        print(f"  DP min_dist={md}: fixed-N +-50 acc = {tot_m/tot_r*100:.1f}%  ({tot_m}/{tot_r})")

if __name__ == "__main__":
    main()
