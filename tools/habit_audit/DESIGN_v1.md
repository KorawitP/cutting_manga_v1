# สถาปัตยกรรมระบบตัดภาพแบบ "ตัดเหมือนผู้ใช้ + รองรับเรื่องใหม่" — Design v1

## หลักการออกแบบ (Core Principle)

**แยกสองชั้นออกจากกัน:**

```
ชั้น 1: "เข้าใจภาพ"  = อะไรคือ gutter, แพเนลอยู่ไหน, ตรงไหนเป็นเนื้อหา
        -> UNIVERSAL (ใช้ได้ทุกเรื่อง/ทุกแนว) -> เทรนจาก data ทั้งหมด -> ถ่ายโอนได้ฟรี

ชั้น 2: "ตัดสินใจ"   = จุดนี้ผมจะตัดไหม / รวมแพเนลไหม / บังคับตัดไหม
        -> PERSONAL (นิสัยผู้ใช้) -> feature-based (ไม่ใช่ pixel) -> เทรนจาก 14k จุดตัดจริง
```

**ทำไมแยก:**
- เรื่องใหม่มา → ชั้น 1 ทำงานได้ทันที (panel/gutter detection เป็นสากล)
- นิสัยของคุณถูกเข้ารหัสเป็น **feature/พารามิเตอร์** (segment length, gutter width, offset, merge rate)
  ซึ่งไม่ผูกกับลายเส้นของเรื่องใดเรื่องหนึ่ง → โอนย้ายได้ + ปรับ/อัปเดตเมื่อนิสัยเปลี่ยนได้ง่าย
- เป็น interpretable: ถอดกฎของคุณออกมาเป็นตัวเลขให้ตรวจเองได้

---

## สถาปัตยกรรม (3 ขั้นตอน)

```
   แถบยาวทั้งตอน (merged strip)
        │
        ▼
┌───────────────────────────────┐
│ STAGE 1: Universal Boundary Map │  ← CNN, เทรนจากทุกเรื่อง
│  ต่อแถว (per-row) ให้:          │
│   • P(boundary)  = โอกาสเป็นจุดตัด  │
│   • is_gutter    = อยู่ใน gutter ไหม │
│   • content_density = ความแน่นเนื้อหา │
└───────────────────────────────┘
        │  ส่งออก candidate positions + feature vector ต่อจุด
        ▼
┌───────────────────────────────┐
│ STAGE 2: Personal Policy "จะตัดไหม"│  ← โมเดลเล็ก feature-based
│  feature ต่อ candidate:         │
│   • segment_height (ยาวแค่ไหนตั้งแต่ตัดล่าสุด)  │
│   • gutter_width / is_gutter    │
│   • content_density รอบจุด      │
│   • distance_to_text/bubble     │
│   • offset_preference           │
└───────────────────────────────┘
        │  accept/reject + rank
        ▼
┌───────────────────────────────┐
│ STAGE 3: Sequential Policy     │  ← กฎ + constraint จากนิสัยคุณ
│  • min segment height          │  (อย่าตัดแพเนลเล็กเกิน)
│  • max segment height          │  (สกัดจาก data: ~p95 ของคุณ)
│  • min distance ระหว่างจุดตัด   │
│  • offset snap (ตัดที่ออฟเซ็ตที่คุณชอบใน gutter) │
└───────────────────────────────┘
        │
        ▼
   จุดตัดสุดท้าย
```

---

## STAGE 1: Universal Boundary Map (CNN)

### โจทย์
ต่อแถวพิกเซล (row) ของทั้งคอลัมน์ ทาย 3 ค่า พร้อมกัน (multi-task head):
- `P(boundary)` — แถวนี้เป็นเส้นแบ่งแพเนลไหม
- `gutter_prob` — แถวนี้อยู่ใน gutter (แถบว่าง) ไหม
- `content_density` — แถวนี้เนื้อหาหนาแน่นแค่ไหน (ใช้เลือกจุดบังคับตัด)

### ข้อมูลเทรน (label จากโครงสร้างไฟล์ ฟรี ไม่ต้องคนทำ)
- **Positive boundary:** รอยต่อระหว่างไฟล์ (14,333 จุด) — ทั้ง clean และ hard case
- **Negative boundary:** interior gutter ที่ผู้ใช้เลือกไม่ตัด (≈575 จุด/100 slice) + แถวเนื้อหาสุ่ม
- **Gutter label:** จาก row-feature detection (std/white/edge) ที่ validated แล้ว

### Architecture
- Encoder เบาๆ (MobileNetV3 / EfficientNet-Lite) รับ input ความกว้างคงที่ (เช่น 512) สูงเต็มคอลัมน์
- 1D head (global-pool แกนกว้าง → เวกเตอร์ [H]) — เห็นบริบททั้งภาพ ไม่ใช่แค่ strip 256px
- หรือ U-Net เล็กๆ ถ้าต้องการความละเอียดต่อจุด
- Augment: synthetic gutter (แทรกแถบว่างระหว่าง content) + flip

### ทำไม universal
Panel/gutter boundary เป็นคุณสมบัติของเว็ปตูนทุกเรื่อง (แถบว่างกั้นแพเนล)
→ เทรนจาก 5 เรื่อง 211 ตอน แล้วใช้กับเรื่องใหม่ได้เลย ไม่ต้อง retrain

---

## STAGE 2: Personal Policy "ผมจะตัดไหม" (โมเดลเล็ก)

### โจทย์
รับ candidate + feature vector → ตัดสินใจ accept/reject
**เรียนรู้ 2 โหมดจากผู้ใช้:**
- **Normal:** มี gutter + segment ยาวพอ → ตัด
- **Hard case:** ไม่มี gutter + segment ยาวเกิน tolerance → บังคับตัดที่จุด "แย่น้อยสุด"

### Feature (สำคัญ: ไม่ใช่ pixel → ไม่ผูกกับเรื่อง/ลายเส้น)
| feature | วิธีวัด | สื่อถึงนิสัยอะไร |
|---|---|---|
| `segment_h` | ความสูงจากจุดตัดก่อนหน้าถึงจุดนี้ | "ยาวแค่ไหนเริ่มทนไม่ไหว" |
| `is_gutter` / `gutter_w` | จาก Stage 1 | "ตัดที่แถบว่างเป็นหลัก" |
| `content_density` | จาก Stage 1 รอบ ±20px | "บังคับตัดที่จุดเนื้อหาน้อยที่สุด" |
| `text_prox` | ระยะถึงตัวอักษร/ฟอง | "ไม่ตัดโดนตัวอักษร/ฟอง" |
| `offset` | ตำแหน่งใน gutter | "ชอบตัดชิดบน/กลาง/ล่าง" |
| `merged_so_far` | จำนวนแพเนลที่รวมใน segment นี้ | "นิสัยรวมแพเนล" |

### Model
- GradientBoosting / Logistic Regression / MLP เล็กๆ
- **เทรนจาก:** positive = 14,333 seams, negative = interior gutter ที่คุณไม่ตัด + candidate จำลอง
- **SHAP** → ถอดกฎของคุณเป็นข้อความ (interpretable) + ตรวจว่า feature ไหนน้ำหนักสุด

### ทำไม personal แต่โอนย้ายได้
เพราะ feature เป็นสากล (segment height เป็นพิกเซล ไม่ขึ้นกับลายเส้น)
→ ฝึกจากนิสัยคุณ แล้วใช้กับเรื่องใหม่ได้

---

## STAGE 3: Sequential Policy (กฎจากนิสัย + constraint)

เดินจากบนลงล่าง (greedy / DP):
1. สะสม segment height
2. ถ้าเจอ gutter candidate + segment ≥ min → ตัด (snap offset)
3. ถ้า segment เกิน `max_segment` (สกัดจาก data ของคุณ) + ยังไม่มีจุดตัด → บังคับตัดที่ candidate
   ที่ content_density ต่ำสุด (เหมือนคุณทำในเคสยาก)
4. บังคับ min distance ระหว่างจุดตัด + min segment height (อย่าตัดแพเนลจิ๋ว)

พารามิเตอร์ (`max_segment`, `min_segment`, offset bias, merge rate) **สกัดจาก data โดยตรง**:
- จาก habit_report: slice height p95 ≈ 6,488px, median 3,684px → max_segment ลองที่ ~5,500–7,000px
- merge rate 78.5% → เกณฑ์การรวมแพเนล

---

## กลยุทธ์การเทรน (Training Strategy)

### ขั้น 0 — Baseline เชิงกฎ (ทำก่อน ไม่ต้อง ML)
- ใช้สถิติจาก data ตั้งพารามิเตอร์ (max/min segment, offset) + gutter detector ง่ายๆ
- วัดผลด้วย "user acceptance" → ได้ baseline เป็นตัวเปรียบเทียบ

### ขั้น 1 — Train Stage 1 (Universal)
- เทรน boundary map จาก label ที่สร้างจากโครงสร้างไฟล์
- ประเมินด้วย boundary IoU / localization error (ไม่ใช่ accuracy)

### ขั้น 2 — Train Stage 2 (Personal)
- เก็บ feature ของทุก candidate (ตัดจริง vs ไม่ตัด) → เทรน policy + SHAP
- ถอดกฎออกมาให้ผู้ใช้ตรวจ ("คุณไม่เคยแยกแพเนลสูงเกิน X" / "คุณชอบตัดห่างขอบล่าง Y px")

### ขั้น 3 — รวม pipeline + post-processing
- เชื่อม Stage 1 → 2 → 3 วัดผลจริงในแอป

---

## การประเมินผล (ต้องวัดสิ่งที่ผู้ใช้รู้สึกจริง)

| ตัวชี้วัด | วิธีวัด | แปลว่า |
|---|---|---|
| **User Acceptance Rate** | % เส้นที่ AI แนะนำแล้วผู้ใช้ไม่ขยับ | "ตัดเหมือนผมจริงไหม" (ตัวชี้วัดหลัก) |
| Localization error | ระยะ |จุดตัดที่ทาย - จุดจริง| px | แม่นแค่ไหน |
| Hard-case coverage | % เคสบังคับตัดที่ AI ทำได้ | เก่งเคสยากไหม |
| Transfer | ทดสอบกับเรื่องที่ไม่อยู่ใน train | ใช้กับเรื่องใหม่ได้ไหม |

---

## ข้อจำกัดที่ต้องยอมรับ (Honest)
1. Stage 2 เทรนจาก 5 เรื่อง → อาจ overfit รสนิยมช่วงเวลานั้น ต้องเก็บ "แก้เส้น" ใหม่เรื่อยๆ (active learning)
2. Hard-case "แย่น้อยสุด" ยังต้องพึ่ง content_density ที่ดี อาจต้องเพิ่ม text detection
3. ตัวเลข user acceptance ต้องผูกกับแอปจริง (วัดจากการที่ผู้ใช้กดยอมรับ/ขยับเส้น)

## ไฟล์ที่วางแผนจะสร้าง
- `core/boundary_model.py` — Stage 1 (CNN)
- `core/personal_policy.py` — Stage 2 (feature + model)
- `core/sequential_policy.py` — Stage 3 (rules)
- `train/train_boundary.py`, `train/train_policy.py`
- `tools/habit_audit/` — ใช้ data ที่สกัดแล้วต่อ
