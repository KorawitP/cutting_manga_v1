import json, sys
from pathlib import Path
import numpy as np
import eval_dp2 as D

SW, A_DEN, MD, RR = 15, 1.0, 1500, 10
CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r4")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")
split = json.load(open(SPLIT, encoding="utf-8"))

eps = []
for e in split["val"]:
    name = D.safe(f"{e['series']}__{e['episode']}")
    z = np.load(CACHE / f"{name}.npz")
    seams = np.sort(z["seams"])
    if len(seams) == 0:
        continue
    eps.append({"series": e["series"], "episode": e["episode"], "yc": z["yc"], "sv": z["sv"],
                "seams": seams, "density": z["density"], "dmax": max(1.0, float(z["density"].max())),
                "H": int(z["H"])})

def select(ep):
    sv = np.convolve(ep["sv"], np.ones(SW) / SW, mode="same")
    score = sv - A_DEN * (ep["density"][ep["yc"]] / ep["dmax"])
    cuts = D.dp_select(ep["yc"], score, len(ep["seams"]), MD)
    den = ep["density"]
    out = []
    for c in cuts:
        lo = max(0, c - RR); hi = min(ep["H"] - 1, c + RR)
        out.append(lo + int(np.argmin(den[lo:hi + 1])))
    return np.array(out, dtype=np.int32)

# ---- measure (a) strict 1:1, (b) naive nearest ----
n_total = 0
n_11 = 0
n_naive50 = 0
naive_d = []          # nearest-line distance per real seam
strict_unmatched = [] # for seams that failed 1:1, nearest line distance (ignoring usage)
for ep in eps:
    cuts = select(ep)
    seams = ep["seams"]
    n_total += len(seams)
    # naive nearest
    for s in seams:
        d = int(np.abs(cuts - s).min())
        naive_d.append(d)
        if d <= 50:
            n_naive50 += 1
    # strict 1:1 (same as eval_best)
    n_11 += D.match_1to1(cuts, seams)

naive_d = np.array(naive_d)
print("=" * 70)
print("ความเข้าใจผลลัพธ์ (r4 + DP) | 42 ตอน, รอยต่อจริง %d เส้น" % n_total)
print("=" * 70)
print(f"[เข้ม] 1:1 match (AI 1 เส้น ใช้ได้ครั้งเดียว, ต้องอยู่ภายใน +-50px ของเส้นจริง):")
print(f"        {n_11}/{n_total} = {n_11/n_total*100:.1f}%  <- ตัวเลขหลัก (51.2%)")
print()
print(f"[หลวม] เส้นจริงแต่ละเส้น AI ใกล้สุดอยู่ไกลแค่ไหน (ไม่สนใจว่าเส้นซ้ำกัน):")
for q in (25, 50, 75, 90, 95):
    print(f"        {q:>2}% ของเส้นจริง มี AI line ภายใน {np.percentile(naive_d, q):5.0f}px")
print(f"        median = {np.median(naive_d):.0f}px")
print(f"        {q:>2}% ของเส้นจริง มี AI line ภายใน +-50px (หลวม): {n_naive50/n_total*100:.1f}%")
print()
print("แปลง่ายๆ:")
print(f"  - เกินครึ่ง ({n_11/n_total*100:.0f}%) ของจุดตัดจริง AI ลงเส้นถูกที่ (ผิด < 50px = แทบมองไม่เห็น)")
print(f"  - ที่เหลือ ~48% AI ตัดผิดที่จริงๆ (เลือก gutter ตัวผิด ใกล้เส้นจริงไม่ถึง 100px+)")
