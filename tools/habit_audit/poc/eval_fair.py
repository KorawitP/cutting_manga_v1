import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import eval_both as E
import joblib

TOL = 150  # user's requirement: +-150px, same for all seams
CACHE_R3 = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r3")

def fair_eval(pred, seams):
    """1:1 matching: each pred line matches at most 1 real seam within +-150px.
    correctness = matched / max(len(pred), len(real))  <- extra lines penalized
    Also report recall & precision for reference."""
    pred = np.sort(pred)
    seams = np.sort(seams)
    matched = 0
    used = set()
    for s in seams:
        d = np.abs(pred - s)
        if len(d) == 0:
            continue
        best = int(np.argmin(d))
        if d[best] <= TOL and best not in used:
            matched += 1
            used.add(best)
    denom = max(len(pred), len(seams))
    return {
        "matched": matched,
        "n_pred": len(pred),
        "n_real": len(seams),
        "correct": matched / max(1, denom) * 100,
        "recall": matched / max(1, len(seams)) * 100,
        "precision": matched / max(1, len(pred)) * 100,
    }

split = json.load(open(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json", encoding="utf-8"))
val = split["val"]
gbm = joblib.load(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\gbm_r1.joblib")

from collections import defaultdict
agg = defaultdict(lambda: {"cnn": None, "gbm": None})
for e in val:
    name = f"{e['series']}__{e['episode']}"
    safe = name.replace("\\", "_").replace("/", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace('"', "_").replace("<", "_").replace(">", "_").replace("|", "_")
    z = np.load(CACHE_R3 / f"{safe}.npz")
    seams = np.sort(z["seams"])

    # CNN tuned: th=0.3, smooth=11
    sv_sm = np.convolve(z["sv"], np.ones(11) / 11, mode="same")
    pk, _ = find_peaks(sv_sm, height=0.3, distance=max(1, 30 // E.STRIDE // 2))
    pred_cnn = E.refine_peaks(z["yc"][pk], z["density"], int(z["H"]))
    r_cnn = fair_eval(pred_cnn, seams)

    # GBM tuned: th=0.1, smooth=31
    ep = E.build_features_for_episode(Path(e["images"]))
    feats = E.norm_feats(ep["feats"])
    probs = gbm.predict_proba(feats)[:, 1]
    score = probs.copy()
    score[feats[:, 4] < 5] *= 0.3
    score = np.convolve(score, np.ones(31) / 31, mode="same")
    pk, _ = find_peaks(score, height=0.1, distance=max(1, 30 // 2))
    pred_gbm = E.refine_peaks(pk, ep["density"], len(probs))
    r_gbm = fair_eval(pred_gbm, seams)

    if agg[e["series"]]["cnn"] is None:
        agg[e["series"]]["cnn"] = {"m": 0, "p": 0, "r": 0}
        agg[e["series"]]["gbm"] = {"m": 0, "p": 0, "r": 0}
    agg[e["series"]]["cnn"]["m"] += r_cnn["matched"]
    agg[e["series"]]["cnn"]["p"] += r_cnn["n_pred"]
    agg[e["series"]]["cnn"]["r"] += r_cnn["n_real"]
    agg[e["series"]]["gbm"]["m"] += r_gbm["matched"]
    agg[e["series"]]["gbm"]["p"] += r_gbm["n_pred"]
    agg[e["series"]]["gbm"]["r"] += r_gbm["n_real"]

print("=" * 78)
print("FAIR EVAL: 1:1 line matching within +-150px | correct = matched / max(n_pred, n_real)")
print("=" * 78)
tot = {"cnn": {"m": 0, "p": 0, "r": 0}, "gbm": {"m": 0, "p": 0, "r": 0}}
for s in sorted(agg):
    c, g = agg[s]["cnn"], agg[s]["gbm"]
    for m, a in (("cnn", c), ("gbm", g)):
        tot[m]["m"] += a["m"]; tot[m]["p"] += a["p"]; tot[m]["r"] += a["r"]
    cc = c["m"] / max(1, max(c["p"], c["r"])) * 100
    gg = g["m"] / max(1, max(g["p"], g["r"])) * 100
    print(f"  {s:38s} CNN corr={cc:5.1f}% ({c['m']}/{max(c['p'],c['r'])}) | "
          f"GBM corr={gg:5.1f}% ({g['m']}/{max(g['p'],g['r'])})")

print("-" * 78)
for m, a in (("CNN", tot["cnn"]), ("GBM", tot["gbm"])):
    corr = a["m"] / max(1, max(a["p"], a["r"])) * 100
    rec = a["m"] / max(1, a["r"]) * 100
    prec = a["m"] / max(1, a["p"]) * 100
    print(f"  {m:3s} TOTAL | correct={corr:.1f}% | recall={rec:.1f}% | precision={prec:.1f}% "
          f"| matched={a['m']} pred={a['p']} real={a['r']}")
