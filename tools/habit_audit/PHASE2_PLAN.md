# POC สรุปผล + แผนระยะ 2 (แก้ 3 ช่องโหว่) + แนวคิดจากงานวิจัย

## ส่วนที่ 1: สรุปผล POC (ข้อมูลจริง 211 ตอน / 14,333 จุดตัด)

### วิธีทดสอบ
- Feature 7 ตัวต่อแถว (std, white, edge, is_gutter, gutter_width, content_density, dist_top)
- Classifier: HistGradientBoosting
- Evaluation: **LOSO** (เทรน 4 เรื่อง ทดสอบ 1 เรื่อง) — วัด "จุดตัดจริงที่ทำนายได้ภายใน ±100px (ปกติ) / ±200px (hard case)"

### ผลลัพธ์

| วิธี | recall (LOSO) | precision | หมายเหตุ |
|---|---|---|---|
| **Peaks + ML** (find_peaks บน classifier score) | **37.9%** | 56.2% | ดีสุด |
| Greedy simulation + ML | 28.9% | 45.8% | greedy สะสม error |
| ตัดทุก gutter (baseline ไม่มี ML) | 30.0% | 40.4% | ตัวเทียบ |
| ตัด interval คงที่ | 6.5% | 10.1% | พื้นฐานที่สุด |

### ตัวชี้วัดต่อเรื่อง (peaks)
- ดีสุด: เรื่อง "1" = 71.3% | be-the-actor = 66.4%
- แย่สุด: i-will-raise = 13.6%
- Normal (gutter) recall รวม = ~41% | **Hard case recall = ~24%**

### ข้อค้นพบ 3 ข้อ (เป็นที่มาของแผนระยะ 2)

1. **Classifier เก่งมาก (AUC 0.94)** — แยก "จุดตัด vs ไม่ตัด" ได้ดี แต่แปลงเป็นพิกัดจริงยังไม่แม่น
2. **Hard case ฉุดทั้งระบบ** — ตัดโดนเนื้อหา (ไม่มี gutter) ทำนายได้แค่ ~24% เพราะเราให้ weight "ตัดใน gutter" เป็นหลัก
3. **Transfer ล้มเหลวบางเรื่อง** — feature scale ต่างกันมากระหว่างเรื่อง (density 13–28, white 0.31–0.53) → classifier เทรนจากเรื่องอื่นใช้กับเรื่องใหม่ไม่ได้ = covariate shift

### บทสรุป POC
- ✅ ML (peaks) **ดีกว่า rule-based** (+8%) — แนวคิดถูกต้องมีศักยภาพ
- ❌ ยังห่างเป้า 80% (ตอนนี้ 37.9%) — ไม่ใช่เพราะแนวคิดผิด แต่เพราะ 3 ช่องโหว่ด้านล่าง

---

## ส่วนที่ 2: แผนระยะ 2 — แก้ 3 ช่องโหว่

### ช่องโหว่ A: Hard case (ตัดโดนเนื้อหา) — recall 24%
**แนวคิดหลัก:** "ตัดที่ gutter" ใช้ไม่ได้กับเคสนี้ ต้องเปลี่ยนเป็น **"หาแถวที่เสียหายน้อยสุด"** (least-damage cut)

**วิธีแก้ (เรียงตามความง่าย→ยาก):**
1. **Cost-based objective แทน classification** — แทนที่จะให้โมเดลทาย "ตัดไหม" ให้ทาย **"แถวนี้เสียหายแค่ไหนถ้าตัด"** (cut damage score) จาก: จำนวน edge ที่โดนตัด, ความสม่ำเสมอสี, ระยะห่าง object ที่ใกล้สุด → ตัดที่แถว cost ต่ำสุด (แรงบันดาลใจ: TransNetV2 "cut at frame of minimal similarity", Video2Comics content-aware)
2. **Segment-length prior** — ถ้าเกิน max_segment (สกัดจาก data) → บังคับตัดที่ cost ต่ำสุดในหน้าต่าง (เลียนแบบนิสัยคุณ 2 โหมด)
3. **Fuse projection profile** — โปรเจกต์ ink ลงแกนแนวตั้ง (classical) = global evidence ช่วยเช็ค "แถวนี้มีเส้นแนวนอนขวางจริงไหม" (Kumiko / X-Y cut)

**งานวิจัยที่อ้างอิง:** TransNet V2 (video shot boundary), Video2Comics content-aware layout, "Robust Panel Extraction for Manga" (Pang 2014)

---

### ช่องโหว่ B: Transfer ข้ามเรื่อง (covariate shift) — fold ตก 71%→13%
**แนวคิดหลัก:** ทำ feature ให้ "ไม่ขึ้นกับลายเส้น" + ปรับต่อเรื่องใหม่

**วิธีแก้ Tier 1 (ทำสัปดาห์นี้ ไม่ต้อง GPU):**
1. **Per-series normalization ที่ test-time** — คำนวณ mean/std ของ feature จากแถวของเรื่องใหม่ (unlabeled) ก่อน predict (Prediction-time Batch Norm, Nado 2020) — ตรงจุดที่สุด
2. **Style randomization ตอนเทรน** — perturb feature statistics ต่อ batch (DARNet channel statistics disruption) → โมเดลไม่ยึดติดช่วงค่าของเรื่องใดเรื่องหนึ่ง
3. **Model soup** — เทรนจากหลาย seed/หลาย split แล้ว average checkpoint
4. **Quantile scaling / log-transform** — edge density เป็น heavy-tailed ต้องแปลงก่อน

**วิธีแก้ Tier 2 (มี label นิดหน่อย):**
5. **RePRI-style transductive** — เทรน 4 เรื่อง แล้ว refine บน unlabeled rows ของเรื่องใหม่ด้วย entropy + KL (เทรน linear head + entropy min) — ไม่ต้อง GPU
6. **DANN gradient-reversal** — เพิ่ม discriminator "มาจากเรื่องไหน" บน features → บังคับให้ feature ใช้ร่วมกันได้ทุกเรื่อง

**ตัวเลือกที่แรงสุด (ถ้ายอมลงทุน):**
7. **สลับ feature ธรรมดา → frozen DINOv2/SAM embedding** — features จาก foundation model โอนย้ายข้ามโดเมนได้ดีมาก (training-free CD-FSS) — แต่ต้องใช้ GPU 1 ครั้ง

**งานวิจัยที่อ้างอิง:** Prediction-time BN (Nado 2020), α-BN (You 2021), DARNet (2023), RePRI (Boudiaf 2021), DANN (Ganin 2016), RobustNet instance whitening (Choi 2021), CD-FSS survey

---

### ช่องโหว่ C: Integration (คะแนน → พิกัดตัด) — recall 37.9%
**แนวคิดหลัก:** find_peaks เป็น local + greedy → error สะสม เปลี่ยนเป็น **global optimization**

**วิธีแก้:**
1. **ruptures / change-point detection (PELT)** — ให้ per-row P(boundary) → DP แบบ global หาจุดแบ่งทั้งหมดพร้อมกัน (ไม่มี error สะสม, ควบคุมจำนวนจุดด้วย penalty) — `pip install ruptures` ~5 บรรทัด
2. **Viterbi / HMM decoding** — state = {ในแพเนล, เส้นแบ่ง}, emission = classifier logits, transition = min-height prior → decode ทั้งคอลัมน์ — มีโครงสร้าง "แพเนลสูงขั้นต่ำ" ในตัว
3. **Parabolic / local-min refinement** — หลังได้ตำแหน่งคร่าว → refine ภายใน ±20px หา local min ของ cost (แม่นขึ้นจาก ±100 → ±10px)
4. **Hybrid 2-pass (แนะนำทำ):** Pass A = classical projection/X-Y-cut หา candidate (global, robust) → Pass B = classifier ให้คะแนน candidate + DP เลือกชุดที่ maximize — ตัดกันทั้ง local error และ accumulation

**งานวิจัยที่อ้างอิง:** Change-point detection (Truong 2020), Text line DP segmentation (Liwicki 2007), Seam carving line segmentation (Saabni 2011), HMM text line (Luthy 2007), X-Y tree (Nagy 1984)

---

## ส่วนที่ 3: งานวิจัย/เครื่องมือที่น่าสนใจ (สรุปจาก 4 การค้นหา)

### 🔥 ใกล้เคียงโจทย์เราที่สุด (โค้ดพร้อมใช้)
| เครื่องมือ | ทำอะไร | เอามาใช้ตรงไหน |
|---|---|---|
| **Toon2Data** (github.com/Snowad14/Toon2Data) | สกัดแพเนลจาก webtoon + inpaint | เดียวที่มีทำ webtoon แนวตั้งตรงๆ — Canny→fill→label |
| **kumiko** (github.com/njean42/kumiko) | ตัดแพเนลการ์ตูน OpenCV | โค้ด line-segment + gutter width — ดูเทคนิค |
| **WebtoonSlicer** (CriPiffe) | "slice webtoon by panel" | reference เล็กๆ อ่านง่าย |
| **gyupro/bubble-detector** | YOLOv8 detect bubble+panel webtoon | ใช้เป็น secondary signal (gutter ไม่มี bubble) |
| **maxhalford comic segmentation** | Canny→dilate→fill→label ~60 บรรทัด | foundation ของ Toon2Data, MIT |

### 📄 งานวิจัยที่ตอบโจทย์เฉพาะจุด
- **CoSMo (2025)** — Page Stream Segmentation = กรอบงานเดียวกับเรา (classify แต่ละ unit ว่าเป็น boundary ไหม) + มี metric ประเมินแบบ stream
- **Manga109 + MangaSeg** — dataset frame annotations (bbox 500k+) ใช้เป็น label เสริม/เทสต์
- **TransNet V2** — video shot boundary: postprocessing (smoothing+NMS+min-length) = template integration ของเรา
- **ruptures PELT** — change-point detection ทดแทน find_peaks (global, แก้ error สะสม)
- **Prediction-time BN / α-BN** — normalize feature ด้วยสถิติของเรื่องใหม่ตอน test (แก้ covariate shift ตรงๆ)
- **RePRI** — few-shot transductive inference: เทรน 4 เรื่อง + refine เรื่องใหม่ด้วย entropy (ไม่ต้อง GPU)
- **DINOv2/SAM frozen features** — ถ้ายอมลงทุน เปลี่ยน feature ธรรมดาเป็น foundation features = โอนย้ายข้ามสไตล์ได้มาก

### ⚠️ ข้อควรรู้
- **ไม่มีงาน "webtoon boundary detection" ตีพิมพ์** — นี่เป็น research gap → โจทย์เราค่อนข้างใหม่
- เครื่องมือ translate (manga-image-translator, comic-translate) **ไม่ได้ตัดแพเนล** — ทำแค่ bubble/text
- License: kumiko AGPL-3.0, manga-image-translator GPL-3.0 — ดูเรื่อง license ถ้าจะ merge โค้ด
- comic-rack/panels smart-zoom เป็น closed-source — ใช้เป็น UX reference เท่านั้น

---

## ส่วนที่ 4: ลำดับการลงมือ (Roadmap ระยะ 2)

**Step 2.0 — Baseline ใหม่ (สัปดาห์นี้):**
- เปลี่ยน integration: find_peaks → ruptures PELT + local refinement (คาด recall +10–15%)
- Per-series test-time normalization (แก้ transfer)

**Step 2.1 — Hard case:**
- เพิ่ม "cut damage score" เป็น output ที่ 2 ของโมเดล + cost-based decision
- เพิ่ม projection profile เป็น global fallback

**Step 2.2 — Transfer ต่อ:**
- Style randomization + model soup
- (ถ้ามี label ใหม่) RePRI transductive

**Step 2.3 — วัดผลจริง:**
- user acceptance rate (ติดในแอป) — ตัวชี้วัดหลักสุดท้าย

---

## ไฟล์ที่เกี่ยวข้อง
- `tools/habit_audit/poc/prepare_poc_dataset.py` — สกัด feature
- `tools/habit_audit/poc/poc_compare.py` — เทรน + LOSO + เทียบ baseline
- `tools/habit_audit/poc/features/poc_final.json` — ผล
- `tools/habit_audit/INSIGHTS.md` — insight นิสัย 2 โหมด
