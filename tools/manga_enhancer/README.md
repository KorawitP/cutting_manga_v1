# Manga Enhancer MVP

เครื่องมือแยกสำหรับ upscale/enhance ภาพมังฮวา/อนิเมะสี หลังจาก export ภาพจากโปรแกรมหลักแล้ว

## การใช้งาน

โฟลเดอร์:

```powershell
python tools\manga_enhancer\main.py "D:\manga\10"
```

หลายโฟลเดอร์:

```powershell
python tools\manga_enhancer\main.py "D:\manga\10" "D:\manga\11" "D:\manga\12"
```

ใช้ไฟล์ list:

```powershell
python tools\manga_enhancer\main.py --list folders.txt
```

`folders.txt`:

```text
D:\manga\10
D:\manga\11
D:\manga\12
```

ไฟล์เดียว:

```powershell
python tools\manga_enhancer\main.py "D:\manga\10\10_1.jpg"
```

ค่า default จะเขียนทับไฟล์เดิมแบบ in-place โดยสร้างไฟล์ temp ก่อน แล้วแทนที่ไฟล์เดิมเฉพาะเมื่อ upscale สำเร็จ

ถ้าต้องการเก็บต้นฉบับไว้ ให้ใช้ `--keep-originals` เพื่อเขียนลงโฟลเดอร์ `upscaled`:

```powershell
python tools\manga_enhancer\main.py "D:\manga\10" --keep-originals
```

```text
D:\manga\10\
  10_1.jpg
  10_2.jpg
  upscaled\
    10_1.jpg
    10_2.jpg
```

## Engine

MVP นี้เรียกใช้ `realesrgan-ncnn-vulkan.exe` แบบ external process

วางไฟล์ executable ไว้ที่:

```text
tools\manga_enhancer\bin\realesrgan-ncnn-vulkan.exe
```

หรือเพิ่ม `realesrgan-ncnn-vulkan.exe` เข้า `PATH`

ค่า default:

- model: `realesr-animevideov3`
- scale: `2`
- tile: `256`
- output: ทับไฟล์เดิม
- queue: ทำทีละโฟลเดอร์ ทีละภาพ เพื่อคุม GPU/VRAM

## ตัวเลือกเสริม

```powershell
python tools\manga_enhancer\main.py "D:\manga\10" --scale 4
python tools\manga_enhancer\main.py "D:\manga\10" --keep-originals
python tools\manga_enhancer\main.py "D:\manga\10" --keep-originals --overwrite
python tools\manga_enhancer\main.py "D:\manga\10" --resume
python tools\manga_enhancer\main.py "D:\manga\10" --dry-run
python tools\manga_enhancer\main.py "D:\manga\10" --recursive
```

`--dry-run` ใช้ตรวจรายการไฟล์และ command โดยไม่ต้องมี engine จริง

`--resume` จะใช้ไฟล์ `.manga_enhancer_done.json` ในโฟลเดอร์งานเพื่อข้ามไฟล์ที่เคยทำสำเร็จแล้ว
