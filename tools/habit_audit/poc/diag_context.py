import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r4")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")
SW, A_DEN, MD, RR = 15, 1.0, 1500, 10

def safe(name):
    return (name.replace("\\", "_").replace("/", "_").replace(":", "_")
            .replace("*", "_").replace("?", "_").replace('"', "_")
            .replace("<", "_").replace(">", "_").replace("|", "_"))

def match_map(cuts, seams, tol=50):
    cuts = np.sort(cuts)
    matched = [False] * len(seams)
    used = set()
    for i, s in enumerate(seams):
        d = np.abs(cuts - s)
        if len(d) == 0:
            break
        best = int(np.argmin(d))
        if d[best] <= tol and best not in used:
            matched[i] = True
            used.add(best)
    return np.array(matched)

split = json.load(open(SPLIT, encoding="utf-8"))

buckets = {"<=50 (model เห็นถูก แต่ selection ไม่เลือก)": 0,
           "50-256 (ใน window 512 เดิมอยู่แล้ว)": 0,
           "256-384 (เฉพาะ window 768 จะเห็นคู่แข่ง)": 0,
           "384-768 (768 เห็นด้วย)": 0,
           ">768 (768 ก็ไม่เห็น)": 0}
miss_n = 0
miss_dist_peak = []
miss_dist_cut = []
matched_seam_peak = []

for e in split["val"]:
    name = safe(f"{e['series']}__{e['episode']}")
    z = np.load(CACHE / f"{name}.npz")
    seams = np.sort(z["seams"])
    if len(seams) == 0:
        continue
    sv = np.convolve(z["sv"], np.ones(SW) / SW, mode="same")
    score = sv - A_DEN * (z["density"][z["yc"]] / max(1.0, float(z["density"].max())))
    yc = z["yc"]
    # all local peaks (candidate boundaries the model is confident about)
    pk, _ = find_peaks(score, distance=max(1, 30 // 16 // 2))
    pk_y = yc[pk]
    pk_sv = score[pk]
    # DP selection
    N = len(seams)
    # build cut set via dp (same as eval_best)
    import eval_dp2 as D
    cuts = D.dp_select(yc, score, N, MD)   # returns cut POSITIONS
    den = z["density"]
    out = []
    for c in cuts:
        lo = max(0, c - RR); hi = min(int(z["H"]) - 1, c + RR)
        out.append(lo + int(np.argmin(den[lo:hi + 1])))
    cuts = np.array(out, dtype=np.int32)
    matched = match_map(cuts, seams)
    for i, s in enumerate(seams):
        # nearest local peak
        d_pk = int(np.abs(pk_y - s).min())
        # nearest selected cut
        d_cut = int(np.abs(cuts - s).min())
        if matched[i]:
            matched_seam_peak.append(d_pk)
        else:
            miss_n += 1
            miss_dist_peak.append(d_pk)
            miss_dist_cut.append(d_cut)
            if d_pk <= 50:
                buckets["<=50 (model เห็นถูก แต่ selection ไม่เลือก)"] += 1
            elif d_pk <= 256:
                buckets["50-256 (ใน window 512 เดิมอยู่แล้ว)"] += 1
            elif d_pk <= 384:
                buckets["256-384 (เฉพาะ window 768 จะเห็นคู่แข่ง)"] += 1
            elif d_pk <= 768:
                buckets["384-768 (768 เห็นด้วย)"] += 1
            else:
                buckets[">768 (768 ก็ไม่เห็น)"] += 1

miss_dist_peak = np.array(miss_dist_peak)
miss_dist_cut = np.array(miss_dist_cut)
matched_seam_peak = np.array(matched_seam_peak)

print(f"รอยต่อทั้งหมด = {miss_n + len(matched_seam_peak)} | matched = {len(matched_seam_peak)} "
      f"({len(matched_seam_peak)/(miss_n+len(matched_seam_peak))*100:.1f}%) | missed = {miss_n}")
print("\n--- ระยะจากรอยต่อจริง ถึง 'peak ที่ model คิดว่าเป็นเส้นแบ่งที่ใกล้สุด' ---")
print("  (peak ใกล้สุด = จุดที่ model มองว่าเป็นขอบ; ถ้าห่างมาก = model หาตำแหน่งผิดจริง)")
print(f"  matched seams:  median={np.median(matched_seam_peak):.0f}px, "
      f"ภายใน 50px = {(matched_seam_peak<=50).mean()*100:.1f}%")
print(f"  missed  seams:  median={np.median(miss_dist_peak):.0f}px, "
      f"ภายใน 50px = {(miss_dist_peak<=50).mean()*100:.1f}%")

print("\n--- สรุปสาเหตุของรอยต่อที่พลาด ({}) เส้น ---".format(miss_n))
for k, v in buckets.items():
    pct = v / max(1, miss_n) * 100
    print(f"  {k:45s} {v:5d}  ({pct:4.1f}%)")

print("\n--- ระยะจากรอยต่อที่พลาด ถึง DP cut ที่ใกล้สุด ---")
for q in (25, 50, 75):
    print(f"  p{q} = {np.percentile(miss_dist_cut, q):.0f}px")

print("\n[สรุป] ถ้า peak ใกล้สุดของ seam ที่พลาดอยู่ในช่วง 256-768px = '768 น่าจะช่วย'")
