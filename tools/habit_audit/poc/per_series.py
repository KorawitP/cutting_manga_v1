import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import eval_both as E
import joblib

split = json.load(open(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json", encoding="utf-8"))
val = split["val"]
gbm = joblib.load(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\gbm_r1.joblib")
CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache")

# per-series breakdown for CNN (tuned) and GBM (tuned)
# CNN: load cache, run with th=0.3 smooth=11
def cnn_episode(z):
    sv_sm = np.convolve(z["sv"], np.ones(11) / 11, mode="same")
    dist = max(1, 30 // E.STRIDE // 2)
    pk, _ = find_peaks(sv_sm, height=0.3, distance=dist)
    peaks = z["yc"][pk]
    return E.refine_peaks(peaks, z["density"], int(z["H"]))

def gbm_episode(probs, gw, density, H):
    score = probs.copy()
    score[gw < 5] *= 0.3
    score = np.convolve(score, np.ones(31) / 31, mode="same")
    pk, _ = find_peaks(score, height=0.10, distance=max(1, 30 // 2))
    return E.refine_peaks(pk, density, H)

from collections import defaultdict
agg = defaultdict(lambda: {"cnn": [0, 0], "gbm": [0, 0]})
for e in val:
    name = f"{e['series']}__{e['episode']}"
    safe = name.replace("\\", "_").replace("/", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace('"', "_").replace("<", "_").replace(">", "_").replace("|", "_")
    z = np.load(CACHE / f"{safe}.npz")
    pred_cnn = cnn_episode(z)
    rc = E.eval_episode(pred_cnn, {"seams": z["seams"], "seg": z["seg"]})
    # gbm
    ep = E.build_features_for_episode(Path(e["images"]))
    feats = E.norm_feats(ep["feats"])
    probs = gbm.predict_proba(feats)[:, 1]
    gw = feats[:, 4]
    pred_gbm = gbm_episode(probs, gw, ep["density"], len(probs))
    rg = E.eval_episode(pred_gbm, {"seams": ep["seams"], "seg": ep["seg"]})
    if rc:
        agg[e["series"]]["cnn"][0] += rc[0] * rc[3]
        agg[e["series"]]["cnn"][1] += rc[3]
    if rg:
        agg[e["series"]]["gbm"][0] += rg[0] * rg[3]
        agg[e["series"]]["gbm"][1] += rg[3]

print("per-series recall (tuned):")
for s in sorted(agg):
    a = agg[s]
    print(f"  {s:38s} cnn={a['cnn'][0]/max(1,a['cnn'][1])*100:.1f}%  gbm={a['gbm'][0]/max(1,a['gbm'][1])*100:.1f}%")
