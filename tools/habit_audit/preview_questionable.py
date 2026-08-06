import argparse
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
MIN_GUTTER_BAND = 3
OUT_DIR = ROOT / "tools" / "habit_audit" / "previews"
ZOOM = 400          # px above/below each seam
OUT_W = 640
GRID_COLS = 2


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


def classify_seam(gray_crop, seam_row):
    """Classify a seam crop into a category. gray_crop is the vertical window,
    seam_row is the absolute row index of the cut inside it."""
    h = gray_crop.shape[0]
    if len(gray_crop.shape) == 3:
        gray = cv2.cvtColor(gray_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_crop

    sobel = cv2.addWeighted(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), 1,
                            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3), 1, 0)
    sobel = cv2.convertScaleAbs(sobel).astype(np.float32)
    std = gray.std(axis=1)
    white = (gray >= WHITE_THRESH).mean(axis=1)
    edge = sobel.mean(axis=1)
    flags = ((std < UNIFORM_STD) | (white >= WHITE_RATIO)) & (edge < EDGE_THRESH)

    # narrow neighborhood around the seam only
    lo = max(0, seam_row - 3)
    hi = min(h, seam_row + 4)
    win = slice(lo, hi)
    seam_clean = bool(flags[win].mean() >= 0.6)
    white_at_seam = float(white[win].mean())
    std_at_seam = float(std[win].mean())

    # text/bubble detection strictly around the cut (±12 px), not the whole crop
    tl = max(0, seam_row - 12)
    tr = min(h, seam_row + 13)
    edge_near = edge[tl:tr].mean() if tr > tl else 0.0
    white_near = white[tl:tr].mean() if tr > tl else 0.0
    text_like = bool(edge_near > 18.0 and white_near > 0.35)

    # dark / colored scene: cut is over a dark/colorful area, no clean white band
    dark = bool(white_at_seam < 0.55 and std_at_seam > 12.0)

    if seam_clean:
        return "D_clean", True
    if text_like:
        return "A_text", False
    if dark:
        return "B_dark_scene", False
    return "C_content", False


def panel_context_crop(strip, flags, sy, H, max_h=2600):
    """Crop the FULL panel above + seam gutter + FULL panel below, so no context is missing."""
    s = sy
    if not flags[s]:
        for d in range(1, 6):
            if s - d >= 0 and flags[s - d]:
                s = s - d
                break
            if s + d < H and flags[s + d]:
                s = s + d
                break
    g_start = s
    while g_start > 0 and flags[g_start - 1]:
        g_start -= 1
    g_end = s
    while g_end < H - 1 and flags[g_end + 1]:
        g_end += 1

    a_top = g_start
    while a_top > 0 and not flags[a_top - 1]:
        a_top -= 1
    a_top0 = a_top
    while a_top0 > 0 and flags[a_top0 - 1]:
        a_top0 -= 1

    b_bot = g_end
    while b_bot < H - 1 and not flags[b_bot + 1]:
        b_bot += 1
    b_bot0 = b_bot
    while b_bot0 < H - 1 and flags[b_bot0 + 1]:
        b_bot0 += 1

    top = a_top0
    bot = b_bot0
    if bot - top > max_h:
        top = sy - max_h // 2
        bot = sy + max_h // 2
        if top < 0:
            bot -= top
            top = 0
        if bot > H:
            bot = H
    return strip[top:bot], sy - top, top, bot


def make_episode_preview(args, series_name, ep_name, images_dir, max_slices, only_suspect):
    files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                   key=lambda p: natural_key(p.name))
    if max_slices:
        files = files[:max_slices]
    if not files:
        return None

    imgs = []
    widths = set()
    for p in tqdm(files, desc="loading", leave=False):
        im = imread(p)
        if im is not None:
            imgs.append(im)
            widths.add(im.shape[1])
    if not imgs:
        return None
    w = sorted(widths)[0]
    for i, im in enumerate(imgs):
        if im.shape[1] != w:
            imgs[i] = cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])))
    strip = np.vstack(imgs)
    H, W = strip.shape[:2]
    cum = np.cumsum([im.shape[0] for im in imgs])

    gray_full = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    sobel = cv2.addWeighted(cv2.Sobel(gray_full, cv2.CV_32F, 1, 0, ksize=3), 1,
                            cv2.Sobel(gray_full, cv2.CV_32F, 0, 1, ksize=3), 1, 0)
    sobel = cv2.convertScaleAbs(sobel).astype(np.float32)
    std = gray_full.std(axis=1)
    white = (gray_full >= WHITE_THRESH).mean(axis=1)
    edge = sobel.mean(axis=1)
    flags = ((std < UNIFORM_STD) | (white >= WHITE_RATIO)) & (edge < EDGE_THRESH)

    panels = []
    for i, sy in enumerate(cum[:-1]):
        crop, rel, top, bot = panel_context_crop(strip, flags, sy, H)
        crop = cv2.resize(crop, (OUT_W, max(1, int(crop.shape[0] * OUT_W / crop.shape[1]))))
        scale = crop.shape[0] / (bot - top)
        yy = int(rel * scale)
        cat, clean = classify_seam(crop, int(rel * scale))

        if only_suspect and clean:
            continue

        cv2.line(crop, (0, yy), (crop.shape[1], yy), (0, 0, 255), 3)
        label = f"[{cat}] y={sy}"
        cv2.putText(crop, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        panels.append(crop)

    if not panels:
        return None

    # build grid
    rows = []
    for i in range(0, len(panels), GRID_COLS):
        row = [panels[j] for j in range(i, min(i + GRID_COLS, len(panels)))]
        hh = max(p.shape[0] for p in row)
        row = [cv2.copyMakeBorder(p, 0, hh - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for p in row]
        rows.append(np.hstack(row))
    cols_w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, cols_w - r.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)) for r in rows]
    canvas = np.vstack(rows)

    out = OUT_DIR / f"{series_name}__{ep_name}__bigseams.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(str(out))
    return out, len(panels)


def main():
    ap = argparse.ArgumentParser(description="Big-context preview of questionable cut seams.")
    ap.add_argument("--series-root", type=Path, default=DEFAULT_SERIES_ROOT)
    ap.add_argument("--series", nargs="+", default=["be-the-actor", "black-corporation-joseon", "evolution-from-the-big-tree", "i-will-raise-this-family-to-greatness"])
    ap.add_argument("--episodes-per-series", type=int, default=2)
    ap.add_argument("--max-slices", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="show ALL seams, not just suspect ones")
    args = ap.parse_args()

    done = 0
    for series in args.series:
        ep_dir = args.series_root / series / "episodes"
        if not ep_dir.is_dir():
            print(f"skip {series}: not found")
            continue
        eps = sorted([d for d in ep_dir.iterdir() if d.is_dir()], key=lambda d: natural_key(d.name))[:args.episodes_per_series]
        for ep in eps:
            res = make_episode_preview(args, series, ep.name, ep / "images", args.max_slices, only_suspect=not args.all)
            if res:
                out, n = res
                print(f"  {series} / {ep.name}: {n} seams -> {out}")
                done += 1
    print(f"\nDone. {done} previews written to {OUT_DIR}")


if __name__ == "__main__":
    main()
