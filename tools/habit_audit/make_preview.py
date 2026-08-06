import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERIES_ROOT = Path(r"D:\python\YT-guide\project-data\series")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OUT_DIR = ROOT / "tools" / "habit_audit" / "previews"
ZOOM = 150       # px above/below each seam to show
OUT_W = 360


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"([0-9]+)", text)]


def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-root", type=Path, default=DEFAULT_SERIES_ROOT)
    ap.add_argument("--series", default="be-the-actor")
    ap.add_argument("--episode", default="ตอนที่ 128")
    ap.add_argument("--max-slices", type=int, default=0)
    args = ap.parse_args()

    images_dir = args.series_root / args.series / "episodes" / args.episode / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{images_dir}")
    files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                   key=lambda p: natural_key(p.name))
    if args.max_slices:
        files = files[:args.max_slices]

    imgs = []
    widths = set()
    for p in tqdm(files, desc="loading"):
        im = imread(p)
        if im is not None:
            imgs.append(im)
            widths.add(im.shape[1])
    w = sorted(widths)[0]
    for i, im in enumerate(imgs):
        if im.shape[1] != w:
            imgs[i] = cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])))
    strip = np.vstack(imgs)
    H, W = strip.shape[:2]

    # cumulative offsets -> seam positions
    cum = np.cumsum([im.shape[0] for im in imgs])

    panels = []
    for sy in cum[:-1]:
        top = max(0, sy - ZOOM)
        bot = min(H, sy + ZOOM)
        crop = strip[top:bot]
        crop = cv2.resize(crop, (OUT_W, int(crop.shape[0] * OUT_W / crop.shape[1])))
        # red line at seam
        yline = int((sy - top) * OUT_W / crop.shape[1] * 1)  # after resize vertical scale == same factor
        yline = int((sy - top) / crop.shape[0] * crop.shape[0])  # keep local
        rel = sy - top
        scale_y = crop.shape[0] / (bot - top)
        yy = int(rel * scale_y)
        cv2.line(crop, (0, yy), (crop.shape[1], yy), (0, 0, 255), 2)
        cv2.putText(crop, f"cut y={sy}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        panels.append(crop)

    if not panels:
        return
    grid_w = 3
    rows = []
    for i in range(0, len(panels), grid_w):
        row = np.hstack([panels[j] for j in range(i, min(i + grid_w, len(panels)))])
        rows.append(row)
    canvas = np.vstack(rows)

    out = OUT_DIR / f"{args.series}__{args.episode}__seams.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(str(out))
    print(f"Saved: {out}  ({len(panels)} seams)")


if __name__ == "__main__":
    main()
