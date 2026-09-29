import sys, time, json
from pathlib import Path
sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import eval_both as E
import joblib

split = json.load(open(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json", encoding="utf-8"))
val = split["val"][:3]
gbm = joblib.load(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\gbm_r1.joblib")
cnn = E.load_cnn(Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_r1.pth"))
for e in val:
    ep = E.build_features_for_episode(Path(e["images"]))
    t0 = time.time(); p = E.run_gbm(ep, gbm); t1 = time.time() - t0
    t0 = time.time(); pc = E.run_cnn(ep, cnn); t2 = time.time() - t0
    r1 = E.eval_episode(p, ep); r2 = E.eval_episode(pc, ep)
    print(f"{e['series'][:15]:15s} gbm_rec={r1[0]*100:.0f}% ({t1:.1f}s) | cnn_rec={r2[0]*100:.0f}% ({t2:.1f}s)")
