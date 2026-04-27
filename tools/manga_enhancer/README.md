# Manga Enhancer CLI

เครื่องมือเพิ่มความชัด/upscale ภาพมังฮวา/อนิเมะสีด้วย `realesrgan-ncnn-vulkan.exe`

ออกแบบให้ใช้ต่อจาก `tools\content_filter_cli` ได้ง่าย โดยเอาภาพใน
`_content_filter_result\keep` ไปสร้างผลลัพธ์ใหม่เป็น `keep_enhanced`
แบบไม่ทับไฟล์เดิมเป็นค่าเริ่มต้น

## โครงไฟล์

```text
tools/manga_enhancer/
  run.py                 CLI แบบเลือกเมนู ใช้งานง่าย
  main.py                CLI เดิมแบบ argument/flag สำหรับงาน automation
  engines/
    realesrgan_ncnn.py
  bin/
    realesrgan-ncnn-vulkan.exe
    models/
```

## ใช้งานแบบง่าย

เปิด PowerShell ที่โปรเจกต์:

```powershell
cd D:\python\cutting_manga_v1
python tools\manga_enhancer\run.py
```

จากนั้นเลือกตามเมนู:

```text
1. ผล content_filter_cli (_content_filter_result\keep)
2. โฟลเดอร์ภาพทั่วไป
```

เลือกขอบเขต:

```text
1. โฟลเดอร์ตอนเดียว
2. โฟลเดอร์เรื่องที่มีหลายตอน
```

ถ้าเลือกทั้งเรื่อง ให้ใส่ path โฟลเดอร์เรื่อง เช่น:

```text
D:\ตัดภาพ\A Modern Man Who Got Transmigrated Into the Murim World โจโฉ
```

โปรแกรมจะหาแต่ละตอนที่มี:

```text
_content_filter_result\keep
```

แล้วประมวลผลทีละตอน

## Workflow ร่วมกับ content_filter_cli

ขั้นแรกแยกภาพด้วย content filter:

```powershell
python tools\content_filter_cli\run.py
```

ผลลัพธ์ต่อหนึ่งตอนจะเป็น:

```text
137/
  _content_filter_result/
    keep/
      137_1.jpg
      137_2.jpg
    discard_suggested/
    review/
    predictions.csv
```

จากนั้นรัน enhancer:

```powershell
python tools\manga_enhancer\run.py
```

ค่า default จะสร้าง:

```text
137/
  _content_filter_result/
    keep/
      137_1.jpg
      137_2.jpg
    keep_enhanced/
      137_1.jpg
      137_2.jpg
```

`keep` เดิมจะไม่ถูกแก้ไข ถ้าไม่ได้เลือกโหมด in-place

## ตัวเลือกผลลัพธ์

ในเมนูจะถามวิธีเขียนผลลัพธ์:

```text
1. สร้างโฟลเดอร์ผลลัพธ์ใหม่ ปลอดภัย ไม่ทับต้นฉบับ
2. ทับไฟล์เดิมแบบ in-place
```

ค่า default:

- ถ้าใช้ผลจาก content filter: เขียนไปที่ `keep_enhanced`
- ถ้าใช้โฟลเดอร์ภาพทั่วไป: เขียนไปที่ `upscaled`

แนะนำให้ใช้โหมดสร้างโฟลเดอร์ใหม่ก่อน เพื่อเทียบผลกับต้นฉบับได้ง่าย

## กรณีเคยทำแล้ว

ถ้ามีผลลัพธ์เดิมหรือเคยทำไปแล้ว เมนูจะให้เลือก:

```text
skip       ข้ามของเดิม
overwrite ลบผลเก่าแล้วทำใหม่
resume    ข้ามไฟล์ที่ state บอกว่าทำสำเร็จแล้ว
```

รายละเอียด:

- `skip` เหมาะกับงานทั้งเรื่องที่ไม่อยากทำตอนที่มี `keep_enhanced` แล้ว
- `overwrite` เหมาะเมื่ออยากเปลี่ยน scale/model แล้วสร้างใหม่ทั้งหมด
- `resume` เหมาะเมื่อรอบก่อนหยุดกลางคัน โปรแกรมจะใช้ `.manga_enhancer_done.json`

## ใช้แบบ command line สั้นๆ

ทั้งเรื่องจากผล content filter:

```powershell
python tools\manga_enhancer\run.py --path "D:\manga\Story" --mode story --source content-filter
```

ตอนเดียวจากผล content filter:

```powershell
python tools\manga_enhancer\run.py --path "D:\manga\Story\137" --mode episode --source content-filter
```

ระบุ path ไปที่ `keep` โดยตรงก็ได้:

```powershell
python tools\manga_enhancer\run.py --path "D:\manga\Story\137\_content_filter_result\keep" --mode episode
```

ทำใหม่ทับ `keep_enhanced` เดิม:

```powershell
python tools\manga_enhancer\run.py --path "D:\manga\Story" --mode story --existing overwrite
```

ทำต่อจากรอบที่ค้าง:

```powershell
python tools\manga_enhancer\run.py --path "D:\manga\Story" --mode story --existing resume
```

ตรวจรายการก่อนทำจริง:

```powershell
python tools\manga_enhancer\run.py --path "D:\manga\Story" --mode story --dry-run
```

ใช้กับโฟลเดอร์ภาพทั่วไป:

```powershell
python tools\manga_enhancer\run.py --path "D:\manga\Story\137" --mode episode --source images
```

ทับไฟล์เดิมแบบ in-place:

```powershell
python tools\manga_enhancer\run.py --path "D:\manga\Story\137" --mode episode --source images --in-place
```

## Engine

เครื่องมือนี้เรียกใช้ `realesrgan-ncnn-vulkan.exe` แบบ external process

วางไฟล์ executable ไว้ที่:

```text
tools\manga_enhancer\bin\realesrgan-ncnn-vulkan.exe
```

หรือเพิ่ม `realesrgan-ncnn-vulkan.exe` เข้า `PATH`

ค่า default:

- model: `realesr-animevideov3`
- scale: `2`
- tile: `256`
- queue: ทำทีละโฟลเดอร์ ทีละภาพ เพื่อคุม GPU/VRAM

## CLI เดิมสำหรับ automation

ยังใช้ `main.py` แบบเดิมได้ ถ้าต้องการระบุ flag เอง:

```powershell
python tools\manga_enhancer\main.py "D:\manga\10"
python tools\manga_enhancer\main.py "D:\manga\10" --keep-originals
python tools\manga_enhancer\main.py "D:\manga\10" --keep-originals --overwrite
python tools\manga_enhancer\main.py "D:\manga\10" --resume
python tools\manga_enhancer\main.py "D:\manga\10" --dry-run
python tools\manga_enhancer\main.py "D:\manga\10" --recursive
```

ค่า default ของ `main.py` จะเขียนทับไฟล์เดิมแบบ in-place โดยสร้างไฟล์ temp ก่อน
แล้วแทนที่ไฟล์เดิมเฉพาะเมื่อ upscale สำเร็จ

ถ้าต้องการเก็บต้นฉบับไว้ ให้ใช้ `--keep-originals` เพื่อเขียนลงโฟลเดอร์ `upscaled`

```powershell
python tools\manga_enhancer\main.py "D:\manga\10" --keep-originals
```
