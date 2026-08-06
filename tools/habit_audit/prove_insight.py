import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERIES_ROOT = Path(r"D:\python\YT-guide\project-data\series")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
WHITE_THRESH = 230
WHITE_RATIO = 0.97
UNIFORM_STD = 12.0
EDGE_THRESH = 4.0


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"([0-9]+)", text)]


def imread_gray(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def seam_clean(prev, next_im):
    up = prev[-40:]
    down = next_im[:40]
    if up.shape[1] != down.shape[1]:
        w = min(up.shape[1], down.shape[1])
        up, down = up[:, :w], down[:, :w]
    win = np.vstack([up, down])
    sobel = cv2.addWeighted(cv2.Sobel(win, cv2.CV_32F, 1, 0, ksize=3), 1,
                            cv2.Sobel(win, cv2.CV_32F, 0, 1, ksize=3), 1, 0)
    sobel = cv2.convertScaleAbs(sobel).astype(np.float32)
    std = win.std(axis=1)
    white = (win >= WHITE_THRESH).mean(axis=1)
    edge = sobel.mean(axis=1)
    flags = ((std < UNIFORM_STD) | (white >= WHITE_RATIO)) & (edge < EDGE_THRESH)
    return bool(flags[30:50].mean() >= 0.6)


def main():
    series_root = Path(r"D:\python\YT-guide\project-data\series")
    rows = []  # (segment_height_above, seam_clean)
    for series in sorted([d for d in series_root.iterdir() if d.is_dir() and not d.name.startswith(".")],
                         key=lambda d: natural_key(d.name)):
        ep_dir = series / "episodes"
        if not ep_dir.is_dir():
            continue
        for ep in sorted([d for d in ep_dir.iterdir() if d.is_dir()], key=lambda d: natural_key(d.name)):
            images_dir = ep / "images"
            if not images_dir.is_dir():
                continue
            files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                           key=lambda p: natural_key(p.name))
            prev = None
            for p in files:
                g = imread_gray(p)
                if g is None:
                    continue
                if prev is not None:
                    clean = seam_clean(prev, g)
                    rows.append((prev.shape[0], clean))
                prev = g

    if not rows:
        return
    arr = np.array(rows)
    clean_h = arr[arr[:, 1] == 1, 0]
    content_h = arr[arr[:, 1] == 0, 0]

    def desc(name, a):
        return (f"{name}: n={len(a)} | "
                f"median={np.median(a):.0f}px | mean={np.mean(a):.0f}px | "
                f"p75={np.percentile(a, 75):.0f}px | p90={np.percentile(a, 90):.0f}px | max={a.max():.0f}px")

    print("== Segment height ABOVE the cut, split by cut type ==")
    print(desc("cut in gutter (clean)", clean_h))
    print(desc("cut through content (hard case)", content_h))
    print(f"\nTotal seams: {len(arr)} | clean={len(clean_h)} ({len(clean_h)/len(arr)*100:.1f}%) | "
          f"content={len(content_h)} ({len(content_h)/len(arr)*100:.1f}%)")
    print(f"\nช่องว่าง median: {np.median(clean_h):.0f}px -> {np.median(content_h):.0f}px "
          f"(= {(np.median(content_h)/np.median(clean_h))*100:.0f}% ของความสูงตอนตัดที่ gutter)")


if __name__ == "__main__":
    main()
