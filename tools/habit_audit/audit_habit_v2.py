import argparse
import collections
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERIES_ROOT = Path(r"D:\python\YT-guide\project-data\series")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
WHITE_THRESH = 230
WHITE_RATIO = 0.97
UNIFORM_STD = 12.0          # row std below this => uniform (colored/white/black gutter)
MIN_GUTTER_BAND = 3
EDGE_THRESH = 4.0           # mean gradient below this => no content detail

# local window around each cut seam for validation
WINDOW_UP = 64
WINDOW_DOWN = 64


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


def row_gutter(row):
    if row.std() < UNIFORM_STD:
        return True
    return (row >= WHITE_THRESH).mean() >= WHITE_RATIO


def compute_margins(gray, sobel):
    h = gray.shape[0]
    row_std = gray.std(axis=1)
    row_white = (gray >= WHITE_THRESH).mean(axis=1)
    row_edge = sobel.mean(axis=1)
    flags = (row_std < UNIFORM_STD) | (row_white >= WHITE_RATIO)
    flags &= row_edge < EDGE_THRESH

    top = 0
    while top < h and flags[top]:
        top += 1
    bottom = 0
    while bottom < h and flags[h - 1 - bottom]:
        bottom += 1

    interior = []
    in_band = False
    start = 0
    for y in range(top, h - bottom):
        if flags[y] and not in_band:
            in_band = True
            start = y
        elif not flags[y] and in_band:
            in_band = False
            if y - start >= MIN_GUTTER_BAND:
                interior.append((start, y - 1))
    if in_band and h - bottom - start >= MIN_GUTTER_BAND:
        interior.append((start, h - bottom - 1))

    return {
        "height": h,
        "top": top,
        "bottom": bottom,
        "content_h": h - top - bottom,
        "interior": interior,
    }


def validate_seam(prev_gray, prev_edge, next_gray, next_edge):
    """Take the last WINDOW_UP rows of prev + first WINDOW_DOWN of next.
    The seam is at index WINDOW_UP. Return whether the seam sits inside a gutter band."""
    up = prev_gray[-WINDOW_UP:]
    up_e = prev_edge[-WINDOW_UP:]
    down = next_gray[:WINDOW_DOWN]
    down_e = next_edge[:WINDOW_DOWN]
    if up.shape[1] != down.shape[1]:
        w = min(up.shape[1], down.shape[1])
        up = up[:, :w]
        up_e = up_e[:, :w]
        down = down[:, :w]
        down_e = down_e[:, :w]
    win = np.vstack([up, down])
    win_e = np.vstack([up_e, down_e])
    flags = (win.std(axis=1) < UNIFORM_STD) | ((win >= WHITE_THRESH).mean(axis=1) >= WHITE_RATIO)
    flags &= win_e.mean(axis=1) < EDGE_THRESH

    seam = WINDOW_UP
    if not flags[seam]:
        # fall back: seam might be exactly at a hard edge; allow ±1 row tolerance
        if not flags[seam - 1] and not flags[seam + 1]:
            return {"in_gutter": False, "gutter_span": 0, "offset_in_gutter": None, "win_flags": flags.tolist()}

    band_start = seam
    while band_start > 0 and flags[band_start - 1]:
        band_start -= 1
    band_end = seam
    while band_end < len(flags) - 1 and flags[band_end + 1]:
        band_end += 1
    span = band_end - band_start + 1
    offset = seam - band_start
    return {"in_gutter": bool(flags[seam]), "gutter_span": int(span), "offset_in_gutter": int(offset)}


def process_episode(args):
    series_name, ep_name, images_dir, min_slices = args
    files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                   key=lambda p: natural_key(p.name))
    if len(files) < min_slices:
        return None

    slices = []
    seams = []
    prev_gray = prev_edge = None
    for p in files:
        gray = imread_gray(p)
        if gray is None:
            continue
        sobel = cv2.addWeighted(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            1, cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3), 1, 0)
        sobel = cv2.convertScaleAbs(sobel).astype(np.float32)
        m = compute_margins(gray, sobel)
        m["name"] = p.name
        slices.append(m)

        if prev_gray is not None:
            seams.append(validate_seam(prev_gray, prev_edge, gray, sobel))
        prev_gray, prev_edge = gray, sobel

    return {
        "series": series_name,
        "episode": ep_name,
        "slices": slices,
        "seams": seams,
    }


def main():
    ap = argparse.ArgumentParser(description="Color-agnostic cutting habit audit + cut-seam validation.")
    ap.add_argument("--series-root", type=Path, default=DEFAULT_SERIES_ROOT)
    ap.add_argument("--out", type=Path, default=ROOT / "tools" / "habit_audit" / "habit_report_v2.json")
    ap.add_argument("--min-slices", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-series", type=int, default=0, help="Only first N series (0 = all)")
    ap.add_argument("--limit-episodes", type=int, default=0, help="Only first N episodes per series (0 = all)")
    args = ap.parse_args()

    if not args.series_root.exists():
        raise FileNotFoundError(f"Series root not found: {args.series_root}")

    series_dirs = sorted([d for d in args.series_root.iterdir() if d.is_dir() and not d.name.startswith(".")],
                         key=lambda d: natural_key(d.name))
    if args.limit_series:
        series_dirs = series_dirs[:args.limit_series]

    jobs = []
    for series in series_dirs:
        ep_dir = series / "episodes"
        if not ep_dir.is_dir():
            continue
        eps = sorted([d for d in ep_dir.iterdir() if d.is_dir()], key=lambda d: natural_key(d.name))
        if args.limit_episodes:
            eps = eps[:args.limit_episodes]
        for ep in eps:
            images_dir = ep / "images"
            if images_dir.is_dir():
                jobs.append((series.name, ep.name, images_dir, args.min_slices))

    print(f"Jobs: {len(jobs)} episodes")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in tqdm(ex.map(process_episode, jobs, chunksize=4), total=len(jobs)):
            if r is not None:
                results.append(r)

    records = []
    seam_rows = []
    for ep in results:
        for s in ep["slices"]:
            records.append({
                "series": ep["series"], "episode": ep["episode"], "name": s["name"],
                "height": s["height"], "content_h": s["content_h"],
                "top": s["top"], "bottom": s["bottom"],
                "n_interior": len(s["interior"]),
                "interior_w_max": max((b - a + 1 for a, b in s["interior"]), default=0),
            })
        for b in ep["seams"]:
            seam_rows.append(b)

    n = len(records)
    hs = np.array([r["height"] for r in records])
    ch = np.array([r["content_h"] for r in records])
    tops = np.array([r["top"] for r in records])
    bots = np.array([r["bottom"] for r in records])
    nint = np.array([r["n_interior"] for r in records])

    in_g = np.array([1 if s["in_gutter"] else 0 for s in seam_rows])
    spans = np.array([s["gutter_span"] for s in seam_rows])
    has_off = [s["offset_in_gutter"] for s in seam_rows if s["offset_in_gutter"] is not None]
    offs = np.array(has_off)
    offs_spans = np.array([s["gutter_span"] for s in seam_rows if s["offset_in_gutter"] is not None])

    def pct(x, p):
        return round(float(np.percentile(x, p)), 1) if len(x) else 0

    report = {
        "meta": {
            "series_root": str(args.series_root),
            "series_count": len(series_dirs),
            "episode_count": len(results),
            "slice_count": n,
            "seam_count": len(seam_rows),
        },
        "slice_height_px": {"min": int(hs.min()), "p25": pct(hs, 25), "median": pct(hs, 50), "p75": pct(hs, 75), "p95": pct(hs, 95), "max": int(hs.max())},
        "content_height_px": {"p25": pct(ch, 25), "median": pct(ch, 50), "p75": pct(ch, 75), "p95": pct(ch, 95)},
        "top_margin_px": {"median": pct(tops, 50), "p95": pct(tops, 95), "zero_pct": round(float((tops == 0).mean() * 100), 2)},
        "bottom_margin_px": {"median": pct(bots, 50), "p95": pct(bots, 95), "zero_pct": round(float((bots == 0).mean() * 100), 2)},
        "cut_validation": {
            "seam_in_gutter_pct": round(float(in_g.mean() * 100), 2),
            "seam_in_gutter_pct_strict": round(float((in_g * (spans >= 5)).mean() * 100), 2),
            "gutter_span_px": {"p25": pct(spans, 25), "median": pct(spans, 50), "p75": pct(spans, 75), "p95": pct(spans, 95)},
            "offset_in_gutter_px": {"p25": pct(offs, 25), "median": pct(offs, 50), "p75": pct(offs, 75),
                                    "top_bias_pct": round(float((offs * 2 < offs_spans).mean() * 100), 2) if len(offs) else 0},
        },
        "merge_habit": {
            "slices_with_interior_gutter_pct": round(float((nint > 0).mean() * 100), 2),
            "interior_bands_per_100_slices": round(float(nint.sum() / n * 100), 2),
            "interior_w_p95": pct(np.array([r["interior_w_max"] for r in records]), 95),
        },
        "per_slice_count": n,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n================ HABIT REPORT v2 ================")
    print(f"Series: {report['meta']['series_count']} | Episodes: {report['meta']['episode_count']} | Slices: {n} | Cut seams: {len(seam_rows)}")
    print("\n-- Cut seam validation (the key proof) --")
    print(f"  seams sitting inside a real gutter : {report['cut_validation']['seam_in_gutter_pct']}%")
    print(f"  seams inside gutter span>=5px     : {report['cut_validation']['seam_in_gutter_pct_strict']}%")
    print(f"  combined gutter span at seam       : p25/median/p75 = {report['cut_validation']['gutter_span_px']['p25']} / {report['cut_validation']['gutter_span_px']['median']} / {report['cut_validation']['gutter_span_px']['p75']} px")
    print(f"  offset of seam inside that gutter  : p25/median/p75 = {report['cut_validation']['offset_in_gutter_px']['p25']} / {report['cut_validation']['offset_in_gutter_px']['median']} / {report['cut_validation']['offset_in_gutter_px']['p75']} px")
    print("\n-- Segment / margins --")
    print(f"  slice height  p25/p50/p75/p95 : {report['slice_height_px']['p25']} / {report['slice_height_px']['median']} / {report['slice_height_px']['p75']} / {report['slice_height_px']['p95']} px (max {report['slice_height_px']['max']})")
    print(f"  top margin zero_pct={report['top_margin_px']['zero_pct']}% | bottom margin zero_pct={report['bottom_margin_px']['zero_pct']}%")
    print("\n-- Merge habit --")
    print(f"  slices with interior gutter: {report['merge_habit']['slices_with_interior_gutter_pct']}%")
    print(f"  interior bands / 100 slices : {report['merge_habit']['interior_bands_per_100_slices']}")
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()
