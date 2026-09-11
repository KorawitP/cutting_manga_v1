"""สคริปต์ตัดแบ่งภาพมังฮวาอัตโนมัติ (Batch AI cutter - ขนาน)

รวมทุกหน้าในแต่ละตอนเป็นภาพยาวแนวตั้ง แล้วใช้ AI (v5) หาจุดตัด
แบบเดียวกับต้นฉบับ (merge -> predict -> split) ประมวลผลหลายตอนพร้อมกันได้

โครงสร้าง input : <INPUT>/<เรื่อง>/<chapter_NNN>/<หน้า>.jpg
โครงสร้าง output: <OUTPUT>/<เรื่อง>/<chapter_NNN>/<chapter_NNN>_<แผง>.jpg

ตัวอย่างการรัน:
    .venv\\Scripts\\python.exe auto_cut.py
    .venv\\Scripts\\python.exe auto_cut.py --workers 4
    .venv\\Scripts\\python.exe auto_cut.py --input D:\\dataMH --output D:\\dataMH_out
    .venv\\Scripts\\python.exe auto_cut.py --quality 95 --skip-existing
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool, cpu_count

# ===== รันจากโฟลเดอร์โปรเจกต์เพื่อ import core ได้ =====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.ai_predictor import AIPredictor
from core.image_ops import merge_images_vertically, split_and_save_image
from PIL import Image

DEFAULT_INPUT = r"D:\dataMH"
DEFAULT_OUTPUT = r"D:\dataMH_out"
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "cnn_v5.pth")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# โครงสร้างที่ worker แต่ละกระบวนการจะเก็บเป็น global
_PREDICTOR = None
_CFG = {}


def init_worker(weights, quality, image_format):
    global _PREDICTOR, _CFG
    _CFG = {"quality": quality, "format": image_format}
    p = AIPredictor(weights)
    ok, msg = p.load_model()
    if not ok:
        raise RuntimeError(f"โหลดโมเดลล้มเหลว: {msg}")
    _PREDICTOR = p


def iter_series_chapters(input_root):
    result = []
    for series in sorted(d for d in os.listdir(input_root)
                         if os.path.isdir(os.path.join(input_root, d))):
        series_dir = os.path.join(input_root, series)
        for chapter in sorted(d for d in os.listdir(series_dir)
                              if os.path.isdir(os.path.join(series_dir, d))):
            result.append((series, chapter, os.path.join(series_dir, chapter)))
    return result


def list_pages(chapter_dir):
    pages = []
    for name in sorted(os.listdir(chapter_dir)):
        if name.lower().endswith(IMAGE_EXTS):
            pages.append(os.path.join(chapter_dir, name))
    return pages


def process_chapter(task):
    series, chapter, chapter_dir, output_root, skip_existing = task
    pages = list_pages(chapter_dir)
    if not pages:
        return (series, chapter, 0, "ไม่มีภาพ")

    out_dir = os.path.join(output_root, series)
    final_dir = os.path.join(out_dir, chapter)
    if skip_existing and os.path.isdir(final_dir) and \
            any(f.startswith(chapter + "_") for f in os.listdir(final_dir)):
        return (series, chapter, 0, "ข้าม (มีผลลัพธ์แล้ว)")

    # 1) รวมทุกหน้าเป็นภาพยาว
    try:
        merged = merge_images_vertically(pages)
    except Exception as exc:
        return (series, chapter, -1, f"รวมภาพล้มเหลว: {exc}")
    if merged is None:
        return (series, chapter, -1, "รวมภาพล้มเหลว")

    # 2) หาจุดตัดด้วย AI
    success, cut_ys, msg = _PREDICTOR.predict_cut_points(merged)
    quality = _CFG["quality"]
    image_format = _CFG["format"]

    def _save_one(img, name):
        os.makedirs(final_dir, exist_ok=True)
        img.save(os.path.join(final_dir, f"{name}.{image_format.lower()}"),
                 image_format, quality=quality)

    if not success:
        _save_one(merged, f"{chapter}_1")
        return (series, chapter, 1, f"AI ล้มเหลว ({msg}) -> เก็บทั้งตอน")

    if not cut_ys:
        _save_one(merged, f"{chapter}_1")
        return (series, chapter, 1, "ไม่มีจุดตัด -> เก็บทั้งตอน")

    # 3) ตัดตามจุดตัด
    result = split_and_save_image(
        merged_img=merged,
        h_breakpoints=cut_ys,
        v_breakpoints=[],
        scale_factor=1.0,
        output_dir=out_dir,
        episode_num=chapter,
        source_name=chapter,
        image_format=image_format,
        quality=quality,
    )
    if not result.success:
        return (series, chapter, -1, result.message)
    return (series, chapter, result.total_images,
            f"จุดตัด {len(cut_ys)} -> แผง {result.total_images}")


def main():
    ap = argparse.ArgumentParser(description="Batch AI manga cutter (merge-then-cut, parallel)")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--format", default="JPEG")
    ap.add_argument("--workers", type=int, default=min(4, max(1, cpu_count())))
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.input):
        print(f"ไม่พบโฟลเดอร์ input: {args.input}")
        sys.exit(1)

    chapters = iter_series_chapters(args.input)
    total = len(chapters)
    if total == 0:
        print("ไม่พบตอนใดๆ ใน input")
        sys.exit(1)

    tasks = [(s, c, d, args.output, args.skip_existing) for (s, c, d) in chapters]

    print(f"โมเดล: cnn_v5 (v5 · MobileNetV3-512)")
    print(f"ประมวลผล {total} ตอน ด้วย {args.workers} กระบวนการ")

    total_panels = 0
    done = 0
    errors = 0
    t0 = time.time()

    if args.workers <= 1:
        init_worker(args.weights, args.quality, args.format)
        results = [process_chapter(t) for t in tasks]
    else:
        results = []
        with Pool(args.workers, initializer=init_worker,
                  initargs=(args.weights, args.quality, args.format)) as pool:
            for res in pool.imap_unordered(process_chapter, tasks):
                results.append(res)

    for series, chapter, n, note in results:
        done += 1
        if n > 0:
            total_panels += n
        if n < 0:
            errors += 1
        print(f"[{done}/{total}] {series}/{chapter}: {note}")

    dt = time.time() - t0
    print(f"\nเสร็จสิ้น: {total} ตอน, {total_panels} แผง "
          f"({errors} ตอนผิดพลาด) ใช้เวลา {dt:.1f}s")
    print(f"ผลลัพธ์อยู่ที่: {args.output}")


if __name__ == "__main__":
    main()
