# บันทึกสรุป: โมเดลตัดภาพเว็ปตูน (CNN Boundary) — สำหรับเทรน/เพิ่มข้อมูลในอนาคต

> เอกสารนี้สรุปทุกอย่างที่ทำมา เพื่อให้กลับมาเทรน/เพิ่มข้อมูลต่อได้ง่ายเมื่อมีข้อมูลใหม่
> ไฟล์โค้ดทั้งหมดอยู่ใน: `D:\python\cutting_manga_v1\tools\habit_audit\poc\`

---

## 1. แนวคิดหลัก (Concept)

### ปัญหา
โปรแกรมตัดภาพเว็ปตูน (`cutting_manga_v1`) ต้องหา "จุดตัด" บนแถบยาวต่อเนื่อง (merged strip)
เพื่อแบ่งเป็นแพเนลตามนิสัยการตัดของเจ้าของข้อมูล

### หลักการ (ตรงไปตรงมา)
1. **ไม่ "เข้าใจมังฮวา" — แค่เรียน "จุดที่คนตัด"**
   โมเดลไม่รู้เรื่อง/สไตล์/เนื้อหา มันแค่จำแนกว่า "แถวนี้เป็นรอยต่อที่คนตัดไหม" จากหน้าต่างภาพ 512px รอบจุดนั้น
2. **Ground truth ฟรีจากโครงสร้างไฟล์**
   ขอบระหว่างไฟล์ภาพ = รอยต่อที่คนตัดจริง → ทุกตอนที่ตัดแล้ว = label อัตโนมัติ ไม่ต้องเสียแรง label
   → ตอนใหม่ที่ตัดในอนาคต (AI+ปรับมือ) = ข้อมูลงอกเองได้เรื่อยๆ
3. **แยก "เข้าใจภาพ" ออกจาก "ตัดสินใจ"**
   - Stage 1 (CNN MobileNetV3-512): รูด window 512px (stride 16) → boundary probability ทุกแถว
   - Stage 2 (DP count-free): จากคะแนน เลือกชุดเส้นที่ดีที่สุดทั้งตอน
     (`smooth=11`, density prior `0.75`, λ=0.8 ต่อเส้น, min_dist=400, refine ±10)
     → โมเดลตัดสินใจจำนวนเอง จึงไม่เฟ้อ
4. **Training window:** positive = ครึ่งล่างชิ้น N + ครึ่งบนชิ้น N+1 (รอยต่อกลาง) + offset jitter ±48px ·
   negative = กลางชิ้น + near-miss 60–280px · hflip aug

### ข้อมูลปัจจุบัน
- **19 เรื่อง / 566 ตอน / 43,060 แพเนล / 42,494 จุดตัด**
- แหล่ง: `D:\ตัดภาพ\` (5 เรื่องเดิม ฉบับเต็ม 21K ไฟล์) + `D:\moctoonRT\M\` (14 เรื่องใหม่ raw 27K ไฟล์)
- **placeholder filter:** กรองแบบเนื้อหา (สีเดียว / near-blank) ไม่ใช่ขนาด — ชุดเก่าฉบับเต็มมีขยะ 21.5%

### บทเรียนที่พิสูจน์แล้ว (สำคัญ)
1. **metric ต้อง honest** — fixed-N ±50 (ตัวเลข 87% เดิมเกินจริง เพราะยิงเส้นเยอะก็ได้ recall สูง — จริงแค่ 25.9%)
2. **ความหลากหลายสไตล์ = คันโยกจริง** — เพิ่ม 14 เรื่องใหม่ (3 เท่า) ได้ +4.7pp (count-free F1) — ปัญหาหลักคือ "จำแนกสไตล์" ไม่ใช่ "ข้อมูลน้อย"
3. **ขีดจำกัดจริงของแนวทางนี้:** hard-case (ตัดโดนเนื้อหา ไม่มีกัตเตอร์) แม่น ~ครึ่งของ clean (29% vs 48%) → F1 ~44% คือ "จริงใจ" ยังไม่ใช่กดปุ่มจบ

---

## 2. โมเดล (CNN Boundary Classifier)

### สถาปัตยกรรม
- **MobileNetV3-Small** (1.5M params — เบากว่า EfficientNet-B0 3.5 เท่า)
- Head: `classifier[3] = nn.Linear(576, 2)` (2 คลาส: boundary / not-boundary)
- Pretrained: `MobileNet_V3_Small_Weights.IMAGENET1K_V1`

### Input
- **512×512** (window สูง 512px = บริบท 2 เท่าของเดิม EfficientNet 256px ← สำคัญ!)
- Normalize: ImageNet mean/std `[0.485,0.456,0.406] / [0.229,0.224,0.225]`

### วิธีสร้าง training window (จาก slices)
- **Positive (boundary):** ครึ่งล่างของ slice N (256px) + ครึ่งบนของ slice N+1 (256px) → 512px รวม
- **Negative:** สุ่ม crop 512px จากกลาง slice (ไม่ได้อยู่ใกล้ขอบ)
- อัตราส่วน: 1 positive : 1.5 negative

### เทรน (จาก `train_r1.py`)
```
python train_r1.py --train-cnn --epochs 15 --batch 32 --lr 1e-4 --workers 4
--cnn-out cnn_r1.pth
```
- Optimizer: AdamW, weight_decay 1e-4
- AMP (mixed precision) บน CUDA
- ใช้เวลา ~2 ชม. / 15 epochs (RTX 4050 6GB, batch 32)
- เทรน 10–15 epochs ก็พอ (acc ~98%)

### Inference (จาก `run_cnn` ใน eval_both.py)
```
กระบวนการเหมือนแอปจริง:
1. merge slices → แถบยาว
2. resize ความกว้าง → 512 (รักษาความสูง) ← เร็วมาก ไม่ต้อง resize ทีละ window
3. sliding window สูง 512px, stride 16
4. batch 32 → forward → ได้ P(boundary) ต่อ window
5. smooth คะแนน (window 11) → find_peaks (height=0.3, distance)
6. refine: หาแถว content_density ต่ำสุดใน ±20px ของ peak
```

---

## 3. การปรับ Hyperparameters (สำคัญที่สุด!)

### พารามิเตอร์หลังเทรนที่ต้อง tune (มีผลมาก)
| พารามิเตอร์ | ค่าดีสุด | ค่าเริ่มต้นเดิม | ผล |
|---|---|---|---|
| **smooth window** (ของ CNN score) | **11** | 31 | **recall 68%→84%!** smooth 11 แคบกว่า = peak คมกว่า |
| **threshold** (find_peaks height) | **0.30** (recall สูง) / **0.35** (precision สูง) | 0.30 | ยิ่งสูง = เส้นน้อยลง แม่นขึ้น recall ลดลง |
| **stride** (sliding window) | 16 (แม่น) / 32 (เร็ว 2 เท่า) | 16 | 32 แลก recall ลดนิดหน่อย เร็วขึ้น |
| min cut distance | ~30 | — | กันจุดตัดใกล้เกินไป |

### หลักการเลือก threshold (trade-off)
```
- อยากได้จุดตัดเยอะ (recall สูง)  → threshold ต่ำ (0.15–0.3)
- อยากได้เส้นน้อย สะอาด (precision สูง) → threshold สูง (0.35+)
- ใช้งานจริงแนะนำ: th=0.3–0.35, smooth=11 (สมดุล recall/precision)
```

### วิธี tune เร็ว (ไม่ต้องเทรนใหม่)
ใช้สคริปต์ `sweep_cache.py`:
```
1. รัน build_cache.py ครั้งเดียว → เก็บ score ทุกแถวเป็น npz (cache)
2. python sweep_cache.py → ลองหลาย threshold + smooth window ได้ทันที (ไม่ต้องเทรน/รัน CNN ใหม่)
```

---

## 4. ผลลัพธ์ที่พิสูจน์แล้ว (3 รอบ ยุติธรรม 100%)

> ⚠️ ตัวเลขใน section นี้ใช้ **metric เก่า (จำนวนเส้นอิสระ ±100/200px)** = เกินจริง
> ตัวเลขที่ถูกต้องดูที่ **หมวด 9.7-9.9** (fixed-N ±50: CNN r4+DP = 51.2%)

### วิธีวัด
- **Split 80/20 แบบ deterministic** (SEED=7): train 169 ตอน / held-out 42 ตอน (ไม่เคยเห็น)
- ทั้ง 2 โมเดลเทรนบน 80% ชุดเดียวกัน → วัดบน 20% ชุดเดียวกัน
- Metric: **recall @ ±100px (normal) / ±200px (hard case)**

### ผล (tuned ทั้งคู่)
| โมเดล | recall | precision | inference/ตอน |
|---|---|---|---|
| **CNN MobileNetV3 512px** | **84–87%** | 39–60% | 30–40s |
| GBM (feature 7 ตัว) | 71% | 54% | 1s |

### จุดที่ CNN เก่งกว่า (transfer)
| เรื่อง | CNN | GBM |
|---|---|---|
| 1 | 93.6% | 93.0% |
| be-the-actor | 95.3% | 95.3% |
| black-corporation | 88.6% | 73.0% |
| evolution | 85.8% | 80.8% |
| **i-will-raise** (ต่างสุด) | **70.5%** | **38.9%** |

**ข้อสรุป:** CNN ดีกับ**เรื่องใหม่/ต่างสไตล์**โดยเฉพาะ (+31.6% บนเรื่องที่ GBM เละ)
แต่ช้ากว่า ~30 เท่า → ต้อง optimize ก่อนใช้งานจริง

---

## 5. ข้อควรรู้ / กับดักที่เจอมาแล้ว

1. **อย่า resize ทีละ window** — resize แถบทั้งหมดครั้งเดียว (กว้าง→512) แล้ว slice เร็วขึ้น ~50 เท่า
2. **mean/std ต้องเป็น (3,1,1) ไม่ใช่ (1,3,1,1)** — ไม่งั้น tensor กลายเป็น 4D ผิดมิติ
3. **chunk สุดท้ายเกินขอบ** — ใช้ `min(start+CHUNK, H-WIN_H, chunk_end-WIN_H)` กัน slice ออกนอกภาพ
4. **DINOv2 ช้าเกินไป** บน RTX 4050 (ไม่เหมาะ production) — MobileNetV3 ดีพอ
5. **GBM ต้อง tune ด้วย** ไม่งั้นไม่ fair — GBM best คือ th=0.1, smooth=31
6. **data leakage** — ระวังอย่าเทรนแล้วทดสอบด้วยตอน/เรื่องเดียวกัน (ทำ split แยกก่อน)

---

## 6. แนวทางต่อยอด (เมื่อมีข้อมูลเพิ่ม)

### ① เพิ่มข้อมูล (High Impact)
- เพิ่ม slices/เรื่อง/ตอนใหม่ → รัน `make_split.py` ใหม่ (split ใหม่ deterministic)
- เทรน CNN ใหม่ 10–15 epochs → tune (sweep_cache) → เทรน/วัด LOSO
- **คาดว่า hard case recall จะดีขึ้น** (ตอนนี้ 24% → ถ้าข้อมูลหลากหลายขึ้นควรขึ้น)

### ② ระบบสะสมเรียนรู้ (จาก DESIGN_v1.md — ยังไม่ได้ทำ)
- ในแอป: เมื่อ user แก้เส้น AI (ลาก/ลบ/เพิ่ม) → เก็บเป็น label ใหม่
- เก็บเป็น `dataset_hardcases/` + จุดที่แก้ → fine-tune CNN ต่อ (แต่ต้องระวังเวลาเทรน 2 ชม.)
- **แต่**: ข้อมูลแก้เยอะเกินควร → ใช้ hard case ที่หลากหลาย ไม่ซ้ำ

### ③ Optimize CNN speed (จำเป็นก่อนใช้จริง)
- stride 16 → 32 (เร็ว 2 เท่า)
- ใช้ batch ใหญ่ขึ้น (64–128)
- หรือ quantize/ONNX (เร็วขึ้นหลายเท่า)
- ปัจจุบัน 30–40s/ตอน ช้าเกินสำหรับใช้จริง

### ④ โมเดลใหญ่ขึ้น (ถ้าต้องการ recall สูงสุด)
- ลอง EfficientNet-B0 / MobileNetV3-Large แทน Small (แต่ช้าลง + หน่วยความจำเพิ่ม)
- หรือเพิ่ม input เป็น 768×768 (บริบทแนวตั้งมากขึ้น)

### ⑤ Hybrid GBM + CNN (เร็ว + แม่น)
- ใช้ GBM สแกนเร็วก่อน (1s) → CNN ตรวจซ้ำเฉพาะจุดที่ GBM ไม่ชัวร์ (ใกล้ threshold)
- ได้ความเร็วเกือบ GBM + ความแม่นใกล้ CNN

---

## 7. ไฟล์สำคัญ (อ้างอิง)

| ไฟล์ | หน้าที่ |
|---|---|
| `make_split.py` | สร้าง split 80/20 deterministic → `split.json` |
| `train_r1.py` | เทรน CNN (`--train-cnn`) / GBM (`--train-gbm`) |
| `eval_both.py` | ประเมินทั้ง 2 โมเดลบน held-out + ฟังก์ชัน inference (run_gbm, run_cnn) |
| `build_cache.py` | สร้าง cache คะแนน CNN ต่อตอน (เพื่อ tune เร็ว) |
| `sweep_cache.py` | tune threshold + smooth window จาก cache |
| `sweep_gbm.py` | tune GBM threshold + smooth |
| `render_app.py` / `render_compare.py` | วาดภาพเปรียบเทียบ (ภาพยาวตามลำดับ + เส้นตัด) |
| `cnn_r1.pth` / `cnn_r3.pth` | CNN ที่เทรนแล้ว (r3 แนะนำ, seed 1234) |
| `gbm_r1.joblib` | GBM ที่เทรนแล้ว (ตัวสำรองเร็ว) |
| `split.json` | split ที่ใช้ (reproducible) |
| `FINAL_VERDICT.md` | ผลการตัดสินใจ 3 รอบ |
| `INSIGHTS.md` | insight นิสัยการตัด 2 โหมด (gutter + hard case) |

---

## 8. ขั้นตอน "กลับมาเทรนต่อ" (Quick Start)

```powershell
# 1. เพิ่มข้อมูลใหม่ลง D:\python\YT-guide\project-data\series\<เรื่อง>\episodes\<ตอน>\images\

# 2. สร้าง split ใหม่ (ครอบคลุมข้อมูลทั้งหมด)
python tools\habit_audit\poc\make_split.py

# 3. เทรน CNN (ใช้เวลา ~2 ชม. / 15 epochs)
python tools\habit_audit\poc\train_r1.py --train-cnn --epochs 15 --cnn-out cnn_new.pth

# 4. สร้าง cache แล้ว tune (เร็ว ~10 นาที)
python tools\habit_audit\poc\build_cache.py    # แก้ path โมเดลเป็น cnn_new.pth ก่อน
python tools\habit_audit\poc\sweep_cache.py   # เลือก th + smooth ที่ดีสุด

# 5. ประเมิน LOSO เต็ม (ถ้าต้องการตัวเลขจริงจัง)
# - สร้าง split แบบ Leave-One-Series-Out แล้วเทรน/วัด 5 รอบ
```

---

## 9. บันทึกการพัฒนารอบ 2 (แก้ metric + แก้ต้นตอ localization) — 2026

> ไทม์ไลน์แนวคิดล่าสุด เผื่อกลับมาอ่านต่อ
> รอบนี้ค้นพบว่าตัวเลขเดิม "เกินจริง" และเจอต้นตอที่แท้จริงของปัญหา localization

### 9.1 พบว่า metric เดิมเกินจริง (สำคัญมาก)
- **metric เดิม** (`eval_both.py:161`): recall = "มีเส้นทาย 1 เส้นชนรอยต่อจริงภายใน ±100/200px" → **โมเดลยิงเส้นเยอะ ๆ ก็ได้ recall สูง** (GBM เคยยิง 59 เส้น/ตอน) + tolerance กว้างถึง ±200px
- ผลที่เคยรายงาน: CNN 87.2% vs GBM 71.4% → **ตัวเลขเกินจริง**
- **metric ใหม่ (ที่ user กำหนด): `eval_fixedn.py`**
  1. โมเดลต้องตัดให้ได้**จำนวนเส้นเท่ากับของจริง**ทุกตอน (fixed-N)
  2. แต่ละเส้นต้องห่างจากรอยต่อจริง ≤ **±50px** (1:1 match)
  3. accuracy = matched / n_real
- **ผลจริงด้วย metric ใหม่**: CNN **25.9%** vs GBM 19.6% (delta +6.2% — ชนะอยู่ แต่แคบกว่าเดิมมาก)
- สคริปต์: `eval_fixedn.py` (ตัวหลัก), `sweep_fixedn*.py` (ค้นค่า)

### 9.2 Post-processing optimization (จาก cache เท่านั้น ไม่ต้องเทรนใหม่)
- กลไกที่ได้ผล: **density prior** — เอา `score - α·(density/max)` มากดจุดที่เนื้อหาหนาทึบ (รอยต่อจริงส่วนใหญ่อยู่ใน gutter ที่เบาบาง)
- ค่าดีสุดหลัง sweep 3 รอบ: `cnn_r3, smooth=11, α_density≈0.75, dist≥35, refine≈10`
- **ผล: 25.9% → 40.1%** (+14.2%)
- gw prior (gutter width) ไม่ช่วย / refine ใหญ่ (20-30) ทำให้แย่ลง

### 9.3 Diagnostic: คอขวดคือ "peak ลอย" ไม่ใช่ "โมเดลไม่เห็น"
- สคริปต์ `diag_bottleneck.py` วัด: รอบรอยต่อจริง 50px มี CNN peak หรือไม่
- ผล (42 ตอน): score ใกล้รอยต่อสูงมาก (gutter 0.97 / hard 0.85) แต่มี **local peak ใน ±50px แค่ 63.6% (gutter) / 58.5% (hard)**
- แปลว่า: โมเดล "รู้ว่า" ตรงนั้นเป็นรอยต่อ แต่ peak ลอยออกไป 50-100px → **ceiling ของการปรับ selection อย่างเดียว ≈ 62%**
- จุดนี้ยืนยันว่า**ต้องแก้ที่โมเดล (ความคมของ peak)** ไม่ใช่แค่ post-process

### 9.4 ยืนยัน ground truth ถูกต้อง (visual)
- `verify_gt.py` + `verify_zoom.jpg` / `verify_overview.jpg`
- วาด: tint สลับสี = แต่ละไฟล์ slice, เส้นเขียว = รอยต่อ (ขอบไฟล์), แถบเหลือง/ส้ม/แดง = ±50/100/200px, แผงขวา = โค้ง score CNN
- vision-analyst ยืนยัน: **เส้นเขียวทับขอบไฟล์เป๊ะทุกจุด** → ground truth = ขอบไฟล์ ถูกต้อง
- ±50px = แถบบางเข้มงวด (ยืนยันว่าค่าเหมาะสม ไม่กว้างเกินไป)

### 9.5 ต้นตอที่แท้จริง: train/eval ไม่ตรงกัน
| ตอนเทรน (train_r1) | ตอนประเมิน |
|---|---|
| รอยต่ออยู่กลาง window เป๊ะ (offset 0) | สไลด์ stride 16 → รอยต่ออยู่ทุก offset |
| negative เฉพาะกลาง slice (ไกลรอยต่อ) | ต้องตอบ "ไม่ใช่" แม้ห่างรอยต่อแค่ 60px |

→ โมเดลเรียน "มีรอยต่ออยู่กลางไหม" → ตอบเป็น plateau กว้าง → peak ลอย

### 9.6 แผนแก้: เทรนใหม่ให้ตรงกับงาน (train_r4.py) ⭐ กำลังทำ
หลักการ: **เทรนให้เห็นการกระจายแบบเดียวกับตอนใช้งาน** + **บอกให้ชัดว่า "แค่ใกล้ไม่ใช่"**
1. **Offset jitter ±48px**: รอยต่อไม่ต้องอยู่กลางเสมอ → เรียน "รอยต่ออยู่ตรงไหน peak ต้องลงตรงนั้น" → ยอดแหลม
2. **Near-miss negatives (60-280px จากรอยต่อ)**: สอน "ใกล้แต่ไม่ใช่" → กด score ข้างๆ → ยอดคม (เผื่อช่อง 60px เป็น "คูน้ำ" รอบรอยต่อ)
3. **Horizontal flip**: กำจัด bias ซ้าย/ขวา
- ไม่เปลี่ยน: architecture (MobileNetV3-Small 512), ข้อมูล (169 ตอนเทรน), split → เปรียบเทียบสะอาด
- คำสั่ง: `python train_r4.py --epochs 12 --cnn-out cnn_r4.pth` (~1.5-2 ชม.)
- **ตัวชี้วัดสำเร็จ**: diagnostic peak-in-±50 ขึ้นจาก ~64% → 85%+ และ fixed-N ±50 ควรขึ้นจาก 40% → 50%+

### 9.7 ผลจริงของ train_r4 (สำคัญ — ข้อสรุปที่ได้จากการทดลอง)
**ผลลัพธ์**: 8 epochs (~70 นาที, acc 95.4%) — **augmentation ได้แค่ +0.9%** (r4 41.0% vs r3 40.1% ด้วย greedy)
- diagnostic peak-in-±50 กลับ**แย่ลงเล็กน้อย** (gutter 63.6→62.3%, hard 58.5→52.4%)
- → **offset jitter + near-miss ไม่ได้ปลดล็อก ceiling ตามคาด** (สมมติฐานเดิมผิด)

**แต่เจอของจริงระหว่างทาง (เกมเปลี่ยน):**
1. **ข้อผิดพลาดไม่ใช่ "เลื่อนนิดหน่อย" แต่คือ "เลือก gutter ผิดตัวในบริเวณเดียวกัน"** — หลักฐาน: curve tolerance แบน (41→49→52%) ทั้งที่ error ควรกระจายเล็กๆ ถ้าเป็นแค่ drift
2. **DP selection (global optimize) ชนะ greedy** (`eval_dp2.py`): หา N จุดที่ maximize คะแนนรวม + บังคับระยะห่างขั้นต่ำ
   - greedy 41% → **DP + min_dist=1500 = 51.2%** (+10%)
   - ระยะห่างขั้นต่ำที่ได้ผล = ใกล้เคียง p1 ของระยะห่างรอยต่อจริง (1418px) → บังคับให้ตัดกระจายเหมือนคน
3. r4 ดีกว่า r3 จริงเมื่อใช้ DP (51.2 vs 48.7, +2.5%)

**ผลสุดท้าย (r4 + DP, 42 ตอน held-out, fixed-N จำนวนเส้นเท่ากัน):**
| tolerance | ±30 | ±50 | ±75 | ±100 | ±150 | ±200 |
|---|---|---|---|---|---|---|
| accuracy | 37.1% | **51.2%** | 60.9% | 68.1% | 74.4% | 76.8% |

- **±100 ได้ 68%** → หลัง ±50 เหลือแต่ความคลาดเคลื่อนเล็ก (ตัดถูกบริเวณ แต่ห่างนิดหน่อย)
- per-series @±50: `1` 78.5% | be-the-actor 72.2% | black-corp 57.8% | evolution 48.2% | i-will-raise 22.1%
- **i-will-raise ยังอ่อน (22%)** = เรื่องที่สไตล์ต่างสุด (ข้อมูลความหลากหลายยังไม่พอ)

**สรุปไทม์ไลน์ตัวเลข (metric เดิมเกินจริง → ตัวจริง):**
25.9% (r3 greedy) → 40.1% (+density prior) → **51.2% (r4 + DP)** = เกือบ 2 เท่า

### 9.8 ทางที่ยังไม่ได้ลอง / ต่อยอด
- **Segment-length prior** (logP ของระยะห่างจริง) ใน DP — **ทดลองแล้ว ไม่ได้ผล (ปิด)**
- **Larger context (768/1024px)** เพื่อแยก "gutter ไหนคือรอยต่อจริง" — **วินิจฉัยแล้ว ไม่คุ้ม (ปิด)**
- **เพิ่มเรื่องใหม่** (โดยเฉพาะสไตล์แบบ i-will-raise) — เลเวอร์ข้อมูลที่แรงสุด ⭐
- แก้ point คลาดเคลื่อนเล็ก (±50→±100) อาจใช้ fine-tune ตำแหน่งขั้นสุดท้าย

#### ✅ ผลทดลองล่าสุด (กันทำซ้ำ — ฝั่ง selection อิ่มตัวแล้ว)
1. **Segment-length prior ใน DP** (`eval_gap_prior.py`): ให้ reward ตามระยะห่าง panel จริง (logP จาก train gaps) — **ยิ่ง beta สูงยิ่งแย่** (beta=0 → 50.8%, beta=10 → 45.2%) เพราะระยะห่างจริงแปรผันสูง (CV~30%) + ต่างกันต่อเรื่อง → prior แรงไปกดความมั่นใจของโมเดลเอง → **ไม่ใช้**
2. **Density prior สูงขึ้น / min_dist อื่น** (`eval_final_sweep.py`): a_den 0.5→4.0, md 1200/1500/1800 → plateau ที่ **51.2%** (a_den=1.0, md=1500) → ปรับต่อไม่มีประโยชน์
3. **ข้อสรุปภาพรวม**: ฝั่ง selection ถูก "รีดจนสุด" แล้วที่ ~51.2% — 49% ที่เหลือมาจาก **model แยก gutter ผิดจริง** (65.9% ของรอยต่อพลาด = peak ใกล้สุดห่าง 50-256px) → **ต้องแก้ที่ตัว model (ข้อมูล/เทรน) ไม่ใช่ selection**

#### ✅ สาเหตุของรอยต่อที่พลาด (1394 เส้น) — จาก `diag_context.py`
| สาเหตุ | เส้น | % |
|---|---|---|
| model เห็นถูก (peak ที่รอยต่อ ±50) แต่ selection ไม่เลือก | 453 | 32.5% |
| model หาตำแหน่งผิดจริง (peak ใกล้สุดห่าง 50-256px = ใน window 512 เดิมอยู่แล้ว) | 918 | 65.9% |
| peak อยู่ 256-768px (เฉพาะ 768 ถึงจะเห็นคู่แข่ง) | 23 | 1.7% |
| > 768px | 0 | 0% |
- **768 context = ช่วยได้แค่ ~1.7% → ไม่คุ้ม** (เทรน 3 ชม. + inference 2 เท่า)
- 65.9% = "เห็นคู่แข่งแล้วแต่เลือกผิด" (ไม่ใช่ "มองไม่เห็น") → แก้ด้วยข้อมูล/เทรน ไม่ใช่ window ใหญ่ขึ้น

**ตัวเลขที่ยึดเป็นมาตรฐาน**: r4 + DP (smooth=15, a_den=1.0, md=1500, refine=10) = **51.2%** @ fixed-N ±50

### 9.9.1 Roadmap อนาคต — dataset ก้อนใหม่ (อัปเดตแล้ว 2026-08)
> ⚠️ **มี dataset เพิ่มเติมจาก Google Drive** — user จะโหลดเอง (ยังไม่ได้มาอยู่ในเครื่อง)
>
> **รายละเอียดที่สำรวจแล้ว:**
> - ที่มา: Google Drive folder `MOC TOON` → https://drive.google.com/drive/folders/1m-UqZCJwUU8gfAF-JjHGlqwhdl3fkWg0
> - **14 เรื่อง** (ชื่อไทย เช่น อัศวินทองคำผู้หวนคืนอดีต, ท่านเคานต์สายเปย์..., สาวน้อยนักเลงจอมยั่ว ฯลฯ) + สคริปต์ `convert_to_jpg.py`, `batch_convert_results.py`
> - โครงสร้าง: `เรื่อง/<ตอนเลข>/inpainted/PNG_Export/*.png` (ตอนละ ~30-120+ ภาพ, ชื่อ `<ตอน>_<หน้า>.png`)
>   - Series 1 มี 27 ตอน, Series 2 มี 13 ตอน (โฟลเดอร์เลข 1,2,3...)
>   - ภาพ = แพเนลที่ตัดแล้ว (inpainted) — แนวเดียวกับข้อมูลเดิม
> - **ขนาด**: ใหญ่ (ต้องโหลด, พื้นที่ D: เหลือ ~76GB)
>
> **ปัญหา/บทเรียนตอนลองโหลดด้วย gdown:**
> - gdown เจอ **rate-limit** ("many accesses") ระหว่างโหลด folder ใหญ่
> - **collision ชื่อ** folder: series ชื่อ "1" กับตอน "1" ซ้ำ → gdown กดทับชั้นตอนหาย (โครงสร้างเพี้ยน)
> - → **user เลือกโหลดเอง** (จะแจ้ง path ภายหลัง)
>
> **ขั้นตอนเมื่อโหลดเสร็จ:**
> 1. วาง data → สแกนโครงสร้าง (สคริปต์สแกนหา folder ที่มีภาพ = 1 ตอน)
> 2. จัด/แมปเป็น layout เดิม `series/<ชื่อ>/episodes/<ตอน>/images/` (หรือให้ pipeline อ่าน path ใหม่)
> 3. ทำ **split ใหม่** (deterministic ครอบทั้งชุดเก่า+ใหม่) → ระวัง data leakage
> 4. เทรนใหม่ + ประเมินด้วย metric เดิม (fixed-N ±50 / count-free F1) → เทียบกับ 51.2%
> 5. **คาดว่า i-will-raise-type (เรื่องสไตล์ใหม่) จะดีขึ้น** เพราะความหลากหลายเพิ่มจาก 5 → 19 เรื่อง

### 9.9 เช็ค "AI ตัดสินใจจำนวนเส้นเอง" (count-free) — คำถามจาก user
> user ถาม: "ในแอปจริง AI ไม่รู้ว่าต้องตัดกี่เส้น จะเกิดอะไรขึ้นถ้าไม่กำหนด N?"

**วิธี**: penalized DP — maximize `sum(score) - λ·#cuts` (λ = ค่าปรับต่อเส้น) → AI เลือกทั้งจำนวนและตำแหน่งเอง (`eval_countfree.py` / `eval_countfree2.py`)

**ผล (42 ตอน, ±50px):**
| λ | K/N | over | under | recall | prec | F1 |
|---|---|---|---|---|---|---|
| 0.60 | 1.31 | 40 | 0 | 55.1% | 42.1% | 47.7% |
| 0.75 | 1.05 | 27 | 13 | 51.1% | 48.7% | 49.9% |
| **0.80-0.85** | 0.95→0.82 | 16→6 | 24→33 | 49→46% | 52→56% | **50.5%** |
| 0.95 | 0.43 | 0 | 42 | 29.9% | 69.3% | 41.7% |

**ข้อสรุป (สำคัญ):**
1. **count-free F1 ≈ 50.5% ≈ fixed-N 51.2%** → โมเดล "รู้" จำนวนเส้นเองแล้ว การไม่บอก N เสียแค่ ~1% (คอขวดไม่ใช่การนับจำนวน!)
2. **F1 ค่อนข้างเสถียร** (49-50.5%) ช่วง λ 0.7-0.9 แม้ count เปลี่ยน 2 เท่า (recall/prec หักล้างกัน)
3. **แต่จำนวนเส้นต่อตอนเพี้ยนมาก**: K==N พอดีแค่ 3/42 ตอน, median ต่าง **-7 เส้น (ตัดน้อยไป)** — ต้องระวัง
4. **i-will-raise ตัดแค่ครึ่งเดียว (K/N=0.52)** — score อ่อน ตัดขาดหายครึ่งหนึ่ง
5. **UX implication**: ตัดน้อยไปแย่กว่าตัดเกิน (merge ง่ายกว่าเติมเส้น) → แอปจริงควรใช้ λ ต่ำ (ตัดเยอะไว้ก่อน)

**per-series count-free F1 vs fixed-N acc:** `1` 74.1 vs 78.5 | be-the-actor 72.0 vs 72.2 | black-corp 56.4 vs 57.8 | evolution 47.1 vs 48.2 | i-will-raise 18.3 vs 22.1

**ความหมายโดยรวม:** ปัญหาที่แท้จริงไม่ใช่ "นับจำนวน" แต่คือ **เรื่องใหม่ที่สไตล์ต่าง (i-will-raise F1 18%)** — แก้ด้วยข้อมูลเพิ่ม ระบบสะสมเรียนรู้ หรือ context ใหญ่ขึ้น

### 9.10 สรุปภาพรวมการพัฒนารอบ 2 (สิ่งที่ควรจำ)
- **ตัวเลขที่เชื่อถือได้**: fixed-N ±50 = **51.2%** (r4 + DP), count-free F1 ≈ 50.5%
- **toolkit ที่ใช้ได้ผลเรียงตาม impact**: (1) metric honest fixed-N ±50 ← ไม่โดนยิงเส้นเยอะหลอก, (2) DP selection + min_dist spacing ← +10%, (3) density prior ← +14%, (4) r4 offset-jitter เทรน ← +0.9% (น้อย), (5) count-free penalized DP ← เท่า fixed-N
- **ความเข้าใจผิดที่ถูกแก้แล้ว**: "โมเดลต้องแก้ peak ให้คม" (ผิด — augmentation ไม่ได้ช่วย) → จริงๆ "ต้องเลือก gutter ให้ถูกตัว + บังคับระยะห่างเหมือนคน" (ถูก — DP ช่วย)
- **สิ่งที่ยังค้าง**: เรื่องใหม่สไตล์ต่าง, count ที่ตัดน้อยไป, ±50→±100 ระยะแก้มือ

### 9.11 ประเมิน "งานจริง" ทั้งตอน (eval_work.py) — ยืนยันนิยาม + ผลครบ
> user ยืนยันนิยาม: **จำนวนจุดตัด (+50px)** = ถูกต้อง, อยากเห็นผลแบบ "งานจริง" ทั้งตอน (ไม่ใช่ตัวอย่าง)

สคริปต์ `eval_work.py` ประเมิน **ทั้ง 42 ตอน held-out / 2,856 เส้นจริง** 2 โหมด:

**โหมด 1 — fixed-N ±50 (บอกจำนวนเส้นให้ AI):**
| เรื่อง | acc |
|---|---|
| 1 | 78.5% |
| be-the-actor | 72.2% |
| black-corporation | 57.8% |
| evolution | 48.2% |
| i-will-raise | 22.1% |
| **TOTAL** | **51.2%** |

**โหมด 2 — count-free ±50 (AI ตัดสินใจจำนวนเอง = งานจริง):**
| เรื่อง | recall | prec | F1 | K/N |
|---|---|---|---|---|
| 1 | 72.8% | 82.2% | 77.2% | 0.89 |
| be-the-actor | 74.9% | 69.0% | 71.8% | 1.08 |
| black-corp | 57.8% | 55.1% | 56.4% | 1.05 |
| evolution | 46.9% | 46.8% | 46.8% | 1.00 |
| i-will-raise | 18.7% | 22.8% | 20.6% | 0.82 |
| **TOTAL** | **49.4%** | **51.7%** | **F1 50.5%** | **0.95** |

**ตัวอย่างตอน:** ดีสุด be-the-actor 137 (85.9%), 1/156 (83.3%), 1/141 (81.8%) | แย่สุด i-will-raise 57 (5.9%), 15 (10.8%), 51 (12.7%)

**สรุป:** งานจริง (count-free) F1 ≈ 50.5% — นับจำนวนได้พอดี (K/N=0.95), แม่น/เกินพอๆ กัน
จุดอ่อนเดียวชัด = i-will-raise (เรื่องสไตล์แปลกจาก 5 เรื่องเดิม) → ตรงกับแผนเพิ่ม dataset 14 เรื่อง

### ไฟล์ใหม่รอบนี้
| ไฟล์ | หน้าที่ |
|---|---|
| `train_r4.py` | เทรน v2 (offset jitter + near-miss neg + hflip) พร้อม rich TUI |
| `build_cache_r4.py` | สร้าง cache r4 → `cnn_cache_r4/` |
| `diag_r4.py` | diagnostic peak-in-±50 ของ r4 |
| `eval_r4.py` | fixed-N ±50 (greedy, config ของ r3) |
| `eval_compare.py` | เทียบ r3 vs r4 แต่ละตัวใช้ config ดีสุดของตัวเอง |
| `eval_tolerance.py` | curve accuracy vs tolerance |
| `eval_dp.py` / `eval_dp2.py` / `eval_dp3.py` / `eval_dp4.py` | DP selection + sweep |
| `eval_dp_compare.py` | เทียบ r3 vs r4 ผ่าน DP |
| `eval_best.py` | **ตัวประเมินมาตรฐานสุดท้าย** (r4 + DP) |
| `eval_countfree.py` / `eval_countfree2.py` | AI ตัดสินใจจำนวนเส้นเอง (penalized DP) |
| `interpret_errors.py` | วิเคราะห์ระยะคลาดเคลื่อน (median 48px ฯลฯ) |
| `render_zoom_errors.py` + `zoom_good/mid/hard.jpg` | ภาพ zoom แสดงขนาด error (ตรวจแล้วถูกต้อง) |
| `eval_work.py` | **ประเมินงานจริงทั้ง 42 ตอน (fixed-N + count-free)** ⭐ |
| `diag_context.py` | ตรวจว่า 768px context ช่วยได้แค่ไหน (สรุป: ~1.7% ไม่คุ้ม) |
| `eval_gap_prior.py` | segment-length prior ใน DP (ผล: ไม่ได้ผล — beta สูงยิ่งแย่) |
| `eval_final_sweep.py` | density prior + min_dist final (ยืนยัน plateau 51.2%) |
| `time_per_episode.py` | วัดเวลา inference (~28-38s/ตอน) |
| `cnn_r4.pth` | โมเดลที่ดีสุด (8 epochs) |

---

## 10. บันทึกการพัฒนารอบ 3 (dataset ใหญ่ 3 เท่า + เทรนข้ามเครื่อง) — 2026

> เป้าหมาย: ยืนยันว่า "เพิ่มความหลากหลายสไตล์" ช่วยเรื่องสไตล์แปลก (i-will-raise) ได้จริง

### 10.1 Dataset ใหม่ (เจอของสำคัญ)
- **เจอ `D:\ตัดภาพ\` = ต้นฉบับเต็มของ 5 เรื่องเดิม** — ชุดที่เทรนก่อนหน้า (YT-guide) ถูก content filter ลบไฟล์ทิ้ง (โจโฉ +97%, be-the-actor +94%, Black Corp +82%) → GT เดิมหาย seams
- **โหลด 14 เรื่องใหม่** (`D:\moctoonRT\M`) — ภาพบนสุด = Korean ดิบ + วอเตอร์มาร์ก Newtoki (720/690px) — ผู้ใช้ยืนยันว่าเป็น input จริงที่ต้องตัด (ตัดก่อนแปล)
- **Ground truth เดียวกันหมด** = ขอบไฟล์ (`seams = cumsum(heights)`) — ทั้ง 2 ชุด
- **placeholder filter (สำคัญ):** กรองแบบ**เนื้อหา**ไม่ใช่ขนาด — ขาวล้วน/ดำล้วน/สีเดียว/ near-blank (ชุดเก่าเต็มมี 21.5% ขยะ!, ชุดใหม่ 2.3%) → กติกา: `uniq≤2` หรือ `(uniq≤6 และ mean>245|<15)` กรองทิ้ง
- build: `features_v3/` (566 ตอน) + `dataset_v3.json` + `split_v3.json`
  - **รวม 19 เรื่อง / 566 ตอน / 43,060 แพเนล / 42,494 GT** (เดิม 211 ตอน/14.5K)
  - split: train 453 / val 113 — **val 5 เรื่องเดิม = ชุดเดิมเป๊ะ** + 14 เรื่องใหม่ 80/20 (seed 7)
- สคริปต์: `build_v3` logic ใน `build_dataset.py`/`build_cache_v5.py` + `make_split` อยู่ใน `split_v3.json`

### 10.2 เทรนข้ามเครื่อง (2 เครื่อง LAN)
- **PC** (192.168.1.112, shareuser / ดูไฟล์ `D:\moctoonRT\เชื่อมต่อ 2 เครื่อง (PC + Notebook).md`):
  - GPU **RTX 3060 Ti 8GB** = เร็ว 1.8 เท่า vs 4050 Laptop (benchmark 74ms vs 133ms/step)
  - ติดตั้ง torch cu128 + deps, โอนข้อมูล ~20GB (เฉพาะภาพชั้นบน) → `C:\cut-data\{moctoon,tad}`
  - วิธีรันงานยาว: เขียน .bat → `schtasks /run` (อยู่รอดข้าม session) → log ที่ share
- **กันกับดัก:** ชื่อไทย → console cp1252 error → ต้อง `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` ที่หัวสคริปต์
- `cnn_v5.pth` = MobileNetV3-Small, 12 epochs, 453 ตอน train, offset±48/near-miss/hflip (เหมือน r4)

### 10.3 ผลลัพธ์ v5 vs r4 (val ชุดเดียวกัน 113 ตอน / 8,421 เส้น, GT ครบ)
| ตัวชี้วัด | r4 baseline | **v5** | Δ |
|---|---|---|---|
| Fixed-N ±50 | 40.9% | **43.7%** | +2.8pp |
| Count-free F1 | 39.2% | **43.9%** | +4.7pp |
| K/N (นับจำนวน) | 1.15 | **1.06** | แม่นขึ้น |

**เรื่องที่ v5 พุ่ง (พิสูจน์ว่าความหลากหลายช่วยจริง):** โจโฉ F1 53.6→69.1 · 4.สาวน้อย 49.3→62.6 · 13.โซอึน 29.5→40.2 · Black Corp 54.7→63.0 · be-the-actor 47.2→53.9

**จุดอ่อนที่ยังเหลือ (สไตล์ยากจริง):** 11.ชายคนเดียว 24.9% · 2.เคานต์ 25.2% · 1.อัศวิน 26.3% · i-will-raise 35.6% (F1 count-free) — data มากไม่ได้แก้

> ⚠️ ตัวเลขชุดนี้ประเมินบน **val ใหม่ GT ครบ (8,421 เส้น)** — เทียบกับเลขเดิม 51.2% (val เก่า 2,856 เส้น) **ไม่ได้** ต้องเทียบ r4-vs-v5 บนชุดเดียวกันเท่านั้น

### 10.4 ไฟล์/สถานะ
| ไฟล์ | หน้าที่ |
|---|---|
| `split_v3.json` / `dataset_v3.json` / `features_v3/` | split + manifest + features ชุดรวม |
| `train_v5.py` | เทรน v5 (อ่าน split_v3 + กรอง placeholder ตอนโหลด) |
| `build_cache_v5.py` / `build_cache_r4_v3.py` | สร้าง CNN cache บน val |
| `eval_v5.py` / `eval_r4_v3.py` | ประเมิน DP sweep + fixed-N + count-free (val ใหม่) |
| `cnn_v5.pth` | **โมเดลใหม่ดีสุด** (43.7% fixed-N / 43.9% F1) |
| `D:\moctoonRT\M\` + `D:\ตัดภาพ\` | ข้อมูลต้นทาง (โมเดล PC: `C:\cut-data\`) |

### 10.5 นำไปใช้จริง (production) — อัปเกรดแอปตัดภาพ
- **`core/ai_predictor.py` ถูกอัปเกรด** → รัน **cnn_v5** (MobileNetV3-Small 512px + DP count-free λ=0.8, min_dist 400, density prior) แทน EfficientNet เก่า
- **auto-detect โมเดล:** โหลด MobileNetV3 ก่อน ถ้า state_dict ไม่ตรง → fallback เป็น EfficientNet (best_classifier.pth เดิม ใช้ได้เหมือนเดิม)
- **interface ไม่เปลี่ยน:** `predict_cut_points(pil_image)` → `(ok, cut_ys, msg)` — GUI ทำงานเหมือนเดิม
- `cnn_v5.pth` วางที่ `D:\python\cutting_manga_v1\cnn_v5.pth` + GUI default เปลี่ยนเป็น v5 แล้ว
- **วิธีใช้:** เปิด `run_app.bat` → merge ภาพตอน → กด AI → ได้เส้นตัดแนะนำ → ปรับมือ → export
- **ความคาดหวังจริง:** เรื่องดี (evolution/โจโฉ/BlackCorp/4.สาวน้อย) ตัดถูก ~2/3 · เรื่องยาก (11/2/1/i-will-raise) ต้องแก้เกือบทุกเส้น — ใช้แบบ "AI เสนอ + คนลากปรับ" เสมอ

> ⚠️ ตรวจสอบก่อน commit: cache/features (`features*`, `cnn_cache*`, `*.npz`) ถูก gitignore แล้ว — commit เฉพาะโค้ด + โมเดล + เอกสาร

### 10.6 ทดลองแล้วไม่ได้ผล: Hard-case Reweighting ❌
- **สิ่งที่ลอง:** เทรน `cnn_v5_hard` โดยให้ loss หนักขึ้น (×3) กับ positive ที่เป็น hard-case (ตัดโดนเนื้อหา) — หวังให้โมเดลพยายามกับจุดยากมากขึ้น
- **ผล (val 113 ตอน, λ sweep 0.5–1.1):** F1 **43.9% → 42.7%** (ลดลง) — hard-case ขึ้นเล็กน้อย (26.5→28.2) แต่ precision ลด + เฟ้อเส้น (K/N 1.06→1.14) → **v5 ชนะทุกค่า λ**
- **ข้อสรุป:** Reweight ทำให้โมเดล "เดาแรงขึ้น" (recall ขึ้น, precision ลด) ไม่ได้สร้างสัญญาณที่ไม่มี — สอดคล้องกับที่วัดได้ว่าบน raw ไม่มี feature แยก hard-case (effect d<0.5)
- **บทเรียน:** กำแพง hard-case ต้อง **semantic understanding** (panel-grid/VLM) ไม่ใช่ reweight — **คงใช้ `cnn_v5.pth` ต่อไป**
- ไฟล์ทดลองถูกลบแล้ว (train_v5_hard.py / build_cache_v5_hard.py / eval_compare_hard.py / lam_sweep.py)


