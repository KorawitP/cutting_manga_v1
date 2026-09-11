# ที่มาโมเดล — cutting_manga_v1

## โมเดลเราเอง (เทรนเอง) → โหลดจาก GitHub Release

Release: `cutting-models-v1` (`https://github.com/KorawitP/cutting_manga_v1/releases/tag/cutting-models-v1`)

| ไฟล์ | ขนาด | SHA256 | ใช้ที่ไหน | สายเทรน |
|---|---|---|---|---|
| `cnn_v5.pth` | 5.92MB | `3a8ed93a…eecc04` | โมเดลตัดภาพหลัก — default ใน `ui/main_window.py`, `auto_cut.py`, `core/ai_predictor.py` (v5 path) | ✅ ครบ — `tools/habit_audit/poc/train_v5.py` + `split_v3.json` + `dataset_v3.json` ดู `tools/habit_audit/TRAINING_GUIDE.md` หมวด 10 |
| `best_content_filter.pth` | 15.59MB | `3039184e…17d7cc` | โมเดลกรองภาพ — default ใน `tools/content_filter_cli/run.py` | ⚠️ weights-only — ใน repo มีแค่ inference (`predictor.py`/`run.py`) ไม่มีสคริปต์เทรน/dataset |

วิธีโหลดกลับ: `powershell scripts/download_models_from_release.ps1` (ตรวจ SHA อัตโนมัติ)

## ของ public (ไม่ฝาก Release — โหลดจากต้นทาง)

`tools/manga_enhancer/` เรียก `realesrgan-ncnn-vulkan` เป็น external process (`tools/manga_enhancer/engines/realesrgan_ncnn.py`):

| ไฟล์ | วางที่ | เอาจากไหน |
|---|---|---|
| `realesrgan-ncnn-vulkan.exe` + `vcomp*.dll` | `tools/manga_enhancer/bin/` | release ทางการของ Real-ESRGAN (ncnn vulkan build) — โหลดชุด Windows แล้วแตกไฟล์ exe+จำนวน dll ermin วางตาม path นี้ หรือใส่ exe ใน PATH ก็ได้ |
| `bin/models/*.bin` + `*.param` (`realesrgan-x4plus`, `-anime`, `realesr-animevideov3 x2/x3/x4`) | `tools/manga_enhancer/bin/models/` | มากับแพ็กเกจ ncnn ข้างบน (โฟลเดอร์ models) — model default คือ `realesr-animevideov3` |

## ตั้งใจไม่รวมใน Release

- `tools/habit_audit/poc/cnn_r4.pth` — รุ่นทดลองเก่า ไม่มีโค้ด production เรียก (โดน v5 แทน)
- `tools/habit_audit/poc/cnn_v5.pth` — สำเนาเดียวกับ root `cnn_v5.pth` (SHA เดียวกัน) ไม่ต้องเก็บซ้ำ
- `best_classifier.pth` (root) — legacy EfficientNet ไม่ใช่ default แล้ว (ยังค้างใน `main.spec` รอถอด)
