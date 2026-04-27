# Content Filter CLI

เครื่องมือนี้ใช้โมเดล `best_content_filter.pth` เพื่อแยกภาพหลังตัดเป็น:

```text
keep               ภาพที่ควรเก็บ
discard_suggested  ภาพที่โมเดลมั่นใจว่าควรทิ้ง
review             ภาพที่ยังไม่ชัดเจน ควรตรวจเอง
```

ค่าเริ่มต้นจะใช้ `copy` เท่านั้น ต้นฉบับไม่ถูกลบหรือย้าย

## โครงไฟล์

```text
tools/content_filter_cli/
  run.py
  predictor.py
  README.md
  models/
    best_content_filter.pth
```

## วิธีใช้แบบง่าย

เปิด PowerShell ที่โปรเจกต์:

```powershell
cd D:\python\cutting_manga_v1
python tools\content_filter_cli\run.py
```

จากนั้นเลือก:

```text
1. โฟลเดอร์ตอนเดียว
2. โฟลเดอร์เรื่องที่มีหลายตอน
```

ถ้าเลือกทั้งเรื่อง ให้ใส่ path เช่น:

```text
D:\ตัดภาพ\A Modern Man Who Got Transmigrated Into the Murim World โจโฉ
```

โปรแกรมจะหาโฟลเดอร์ตอนที่มีรูปภาพ เช่น `137`, `138`, `139` แล้วประมวลผลทีละตอน

## Output

ผลลัพธ์จะอยู่ในแต่ละโฟลเดอร์ตอน:

```text
137/
  137_1.jpg
  137_2.jpg
  _content_filter_result/
    keep/
    discard_suggested/
    review/
    predictions.csv
```

`predictions.csv` เก็บคะแนนของแต่ละไฟล์:

```text
file, route, pred_class, discard_prob, keep_prob, error
```

## Existing Result

ถ้าโฟลเดอร์ตอนมี `_content_filter_result` อยู่แล้ว จะมี 3 ทางเลือก:

```text
skip       ข้ามตอนนั้น
overwrite ลบผลเก่าแล้วทำใหม่
append     เพิ่มผลใหม่โดยไม่ลบของเก่า
```

ค่า default คือ `skip`

## Copy / Move

```text
copy  ปลอดภัย ต้นฉบับยังอยู่
move  ย้ายไฟล์จริง
```

ค่า default คือ `copy`

ช่วงแรกควรใช้ `copy` ก่อน และตรวจ `discard_suggested` ก่อนลบทิ้งจริง

## Threshold

ค่า default:

```text
discard_threshold = 0.95
keep_threshold    = 0.80
```

หลักการ:

```text
discard_prob >= 0.95  -> discard_suggested
keep_prob >= 0.80     -> keep
อื่นๆ                 -> review
```

ถ้าต้องการให้ทิ้งน้อยลงและปลอดภัยขึ้น ให้เพิ่ม `discard_threshold` เช่น `0.97`

## ใช้แบบ command line ตรงๆ

ทั้งเรื่อง:

```powershell
python tools\content_filter_cli\run.py --path "D:\ตัดภาพ\A Modern Man Who Got Transmigrated Into the Murim World โจโฉ" --mode story
```

ตอนเดียว:

```powershell
python tools\content_filter_cli\run.py --path "D:\ตัดภาพ\A Modern Man Who Got Transmigrated Into the Murim World โจโฉ\137" --mode episode
```

overwrite ผลเก่า:

```powershell
python tools\content_filter_cli\run.py --path "D:\ตัดภาพ\A Modern Man Who Got Transmigrated Into the Murim World โจโฉ" --mode story --existing overwrite
```

## ข้อควรระวัง

- อย่าใช้ `move` จนกว่าจะมั่นใจ threshold และผลลัพธ์
- โมเดลอาจพลาดภาพสีขาวที่ยังเป็นเนื้อหาจริง ให้ตรวจ `discard_suggested` เป็นระยะ
- ถ้าเจอเคสพลาด ให้เอากลับไปเพิ่มใน dataset แล้วเทรนใหม่
