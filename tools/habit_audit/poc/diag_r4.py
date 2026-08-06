import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r4")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")

def safe(n):
    return (n.replace("\\", "_").replace("/", "_").replace(":", "_")
            .replace("*", "_").replace("?", "_").replace('"', "_")
            .replace("<", "_").replace(">", "_").replace("|", "_"))

split = json.load(open(SPLIT, encoding="utf-8"))
pk_g = pk_h = n_g = n_h = 0
for e in split["val"]:
    name = safe(f"{e['series']}__{e['episode']}")
    z = np.load(CACHE / f"{name}.npz")
    sv = np.convolve(z["sv"], np.ones(11) / 11, mode="same")
    pk, _ = find_peaks(sv, distance=max(1, 30 // 16 // 2))
    pk_y = z["yc"][pk]
    for s, seg in zip(z["seams"], z["seg"]):
        if np.abs(pk_y - s).min() <= 50:
            if seg:
                pk_g += 1
            else:
                pk_h += 1
        if seg:
            n_g += 1
        else:
            n_h += 1
print(f"CNN r4 | peak within +-50px: gutter={pk_g/max(1,n_g)*100:.1f}% ({pk_g}/{n_g})  "
      f"hard={pk_h/max(1,n_h)*100:.1f}% ({pk_h}/{n_h})")
print("  (r3 baseline was: gutter 63.6% / hard 58.5%)")
