import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")

TOL = 50
CACHE_R3 = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r3")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")

def safe(name):
    return (name.replace("\\", "_").replace("/", "_").replace(":", "_")
            .replace("*", "_").replace("?", "_").replace('"', "_")
            .replace("<", "_").replace(">", "_").replace("|", "_"))

split = json.load(open(SPLIT, encoding="utf-8"))
n_hit_g, n_hit_h, n_g, n_h = 0, 0, 0, 0
maxloc_g, maxloc_h = [], []
peakloc_g, peakloc_h = [], []
for e in split["val"]:
    name = safe(f"{e['series']}__{e['episode']}")
    z = np.load(CACHE_R3 / f"{name}.npz")
    sv = np.convolve(z["sv"], np.ones(11) / 11, mode="same")
    yc = z["yc"]
    for s, seg in zip(z["seams"], z["seg"]):
        d = np.abs(yc - s)
        within = d <= TOL
        if not within.any():
            continue
        local = sv[within]
        best_i = int(np.argmax(local))
        m = float(local[best_i])
        if seg:
            n_g += 1; maxloc_g.append(m)
        else:
            n_h += 1; maxloc_h.append(m)
# also count seams with any local peak (find_peaks) inside +-50
n_pk_g, n_pk_h = 0, 0
for e in split["val"]:
    name = safe(f"{e['series']}__{e['episode']}")
    z = np.load(CACHE_R3 / f"{name}.npz")
    sv = np.convolve(z["sv"], np.ones(11) / 11, mode="same")
    yc = z["yc"]
    pk, _ = find_peaks(sv, distance=max(1, 30 // 16 // 2))
    pk_y = yc[pk]
    for s, seg in zip(z["seams"], z["seg"]):
        if np.abs(pk_y - s).min() <= TOL:
            if seg: n_pk_g += 1
            else: n_pk_h += 1

print("seams with ANY model score sampled within +-50px:")
print(f"  gutter(normal): max-score {np.mean(maxloc_g):.3f} (n={n_g})")
print(f"  hard-case     : max-score {np.mean(maxloc_h):.3f} (n={n_h})")
print("seams with a local PEAK within +-50px:")
print(f"  gutter(normal): {n_pk_g}/{n_g} = {n_pk_g/max(1,n_g)*100:.1f}%")
print(f"  hard-case     : {n_pk_h}/{n_h} = {n_pk_h/max(1,n_h)*100:.1f}%")
print("=> if local peak exists near a seam, a SMARTER SELECTION could match it "
      "(upper bound for selection-only improvement)")
