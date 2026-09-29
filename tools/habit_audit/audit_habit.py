import argparse
import collections
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERIES_ROOT = Path(r"D:\python\YT-guide\project-data\series")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
WHITE_THRESH = 230          # pixel >= this is "near white"
WHITE_RATIO = 0.97          # row is "gutter row" if this fraction of pixels is near-white
MIN_GUTTER_BAND = 3         # min row-run to count as an interior gutter band


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"([0-9]+)", text)]


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def row_is_gutter(row):
    return (row >= WHITE_THRESH).mean() >= WHITE_RATIO


def analyze_slice(img):
    h = img.shape[0]
    gutter_flags = np.array([row_is_gutter(img[y]) for y in range(h)], dtype=bool)

    top_white = 0
    while top_white < h and gutter_flags[top_white]:
        top_white += 1
    bottom_white = 0
    while bottom_white < h and gutter_flags[h - 1 - bottom_white]:
        bottom_white += 1

    interior_gutters = []
    in_band = False
    band_start = 0
    for y in range(top_white, h - bottom_white):
        is_g = bool(gutter_flags[y])
        if is_g and not in_band:
            in_band = True
            band_start = y
        elif not is_g and in_band:
            in_band = False
            if y - band_start >= MIN_GUTTER_BAND:
                interior_gutters.append((band_start, y - 1))
    if in_band and h - bottom_white - band_start >= MIN_GUTTER_BAND:
        interior_gutters.append((band_start, h - bottom_white - 1))

    content_h = h - top_white - bottom_white
    return {
        "height": h,
        "content_h": content_h,
        "top_white": top_white,
        "bottom_white": bottom_white,
        "interior_gutters": interior_gutters,
        "edge_content": top_white == 0 or bottom_white == 0,
    }


def collect_episode(images_dir):
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    slices = []
    for p in files:
        img = imread_unicode(p)
        if img is None:
            continue
        info = analyze_slice(img)
        info["name"] = p.name
        slices.append(info)
    return slices


def main():
    ap = argparse.ArgumentParser(description="Extract cutting-habit statistics from manual cut slices.")
    ap.add_argument("--series-root", type=Path, default=DEFAULT_SERIES_ROOT)
    ap.add_argument("--out", type=Path, default=ROOT / "tools" / "habit_audit" / "habit_report.json")
    ap.add_argument("--min-slices", type=int, default=5, help="Skip episodes with fewer slices than this.")
    args = ap.parse_args()

    if not args.series_root.exists():
        raise FileNotFoundError(f"Series root not found: {args.series_root}")

    records = []       # one row per slice
    episode_meta = []  # one row per episode
    skipped = {"too_few": 0, "no_images": 0, "unreadable": 0}

    series_dirs = sorted([d for d in args.series_root.iterdir() if d.is_dir() and not d.name.startswith(".")],
                         key=lambda d: natural_key(d.name))

    for series in tqdm(series_dirs, desc="Series"):
        ep_dir = series / "episodes"
        if not ep_dir.is_dir():
            continue
        for ep in sorted([d for d in ep_dir.iterdir() if d.is_dir()], key=lambda d: natural_key(d.name)):
            images_dir = ep / "images"
            if not images_dir.is_dir():
                continue
            slices = collect_episode(images_dir)
            if not slices:
                skipped["no_images"] += 1
                continue
            if len(slices) < args.min_slices:
                skipped["too_few"] += 1
                continue

            # boundary = between consecutive slices; combined gutter + his offset
            boundaries = []
            for i in range(len(slices) - 1):
                a, b = slices[i], slices[i + 1]
                boundaries.append({
                    "gutter_px": a["bottom_white"] + b["top_white"],
                    "offset_px": a["bottom_white"],       # how deep from the gutter top he cut
                })

            episode_meta.append({
                "series": series.name,
                "episode": ep.name,
                "n_slices": len(slices),
            })
            for s in slices:
                records.append({
                    "series": series.name,
                    "episode": ep.name,
                    "name": s["name"],
                    "height": s["height"],
                    "content_h": s["content_h"],
                    "top_white": s["top_white"],
                    "bottom_white": s["bottom_white"],
                    "n_interior": len(s["interior_gutters"]),
                    "interior_w_max": max((b - a + 1 for a, b in s["interior_gutters"]), default=0),
                    "edge_content": s["edge_content"],
                })

    if not records:
        raise RuntimeError("No usable slices found.")

    heights = np.array([r["height"] for r in records])
    content_hs = np.array([r["content_h"] for r in records])
    tops = np.array([r["top_white"] for r in records])
    bottoms = np.array([r["bottom_white"] for r in records])
    interiors = np.array([r["n_interior"] for r in records])

    all_boundaries = [b for m in episode_meta for b in []]
    gutters = []
    offsets = []
    # boundaries were per-episode; rebuild flat
    idx = 0
    for m in episode_meta:
        for i in range(m["n_slices"] - 1):
            a = records[idx + i]
            b = records[idx + i + 1]
            gutters.append(a["bottom_white"] + b["top_white"])
            offsets.append(a["bottom_white"])
        idx += m["n_slices"]
    gutters = np.array(gutters)
    offsets = np.array(offsets)

    def pct(x, p):
        return round(float(np.percentile(x, p)), 1)

    report = {
        "meta": {
            "series_root": str(args.series_root),
            "series_count": len(series_dirs),
            "episode_count": len(episode_meta),
            "slice_count": len(records),
            "skipped": skipped,
        },
        "slice_height_px": {
            "min": int(heights.min()), "max": int(heights.max()),
            "p25": pct(heights, 25), "median": pct(heights, 50),
            "p75": pct(heights, 75), "p95": pct(heights, 95),
        },
        "content_height_px": {
            "p25": pct(content_hs, 25), "median": pct(content_hs, 50),
            "p75": pct(content_hs, 75), "p95": pct(content_hs, 95),
        },
        "top_margin_px": {"median": pct(tops, 50), "p75": pct(tops, 75), "p95": pct(tops, 95), "zero_pct": round(float((tops == 0).mean() * 100), 2)},
        "bottom_margin_px": {"median": pct(bottoms, 50), "p75": pct(bottoms, 75), "p95": pct(bottoms, 95), "zero_pct": round(float((bottoms == 0).mean() * 100), 2)},
        "cut_gutter_px": {"p25": pct(gutters, 25), "median": pct(gutters, 50), "p75": pct(gutters, 75), "p95": pct(gutters, 95)},
        "cut_offset_px": {"p25": pct(offsets, 25), "median": pct(offsets, 50), "p75": pct(offsets, 75),
                          "top_bias_pct": round(float((offsets < (gutters - offsets)).mean() * 100), 2)},
        "merge_habit": {
            "slices_with_interior_gutter_pct": round(float((interiors > 0).mean() * 100), 2),
            "slices_merged_2plus_pct": round(float((interiors >= 2).mean() * 100), 2),
            "interior_bands_per_100_slices": round(float(interiors.sum() / len(interiors) * 100), 2),
            "interior_w_p95": pct(np.array([r["interior_w_max"] for r in records]), 95),
        },
        "edge_content_pct": round(float(np.array([r["edge_content"] for r in records]).mean() * 100), 2),
        "per_episode": episode_meta,
        "per_slice": records,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n================ HABIT REPORT ================")
    print(f"Series: {report['meta']['series_count']} | Episodes: {report['meta']['episode_count']} | Slices: {report['meta']['slice_count']}")
    print("\n-- Slice height (how long he lets a segment run) --")
    for k in ("min", "p25", "median", "p75", "p95", "max"):
        print(f"  {k:>6}: {report['slice_height_px'][k]} px")
    print("\n-- Content height (excluding gutters) --")
    for k in ("p25", "median", "p75", "p95"):
        print(f"  {k:>6}: {report['content_height_px'][k]} px")
    print("\n-- Cut placement in the gutter --")
    print(f"  gutter width   p25/p50/p75/p95: {report['cut_gutter_px']['p25']} / {report['cut_gutter_px']['median']} / {report['cut_gutter_px']['p75']} / {report['cut_gutter_px']['p95']} px")
    print(f"  cut offset     p25/p50/p75:      {report['cut_offset_px']['p25']} / {report['cut_offset_px']['median']} / {report['cut_offset_px']['p75']} px")
    print(f"  top-biased cuts: {report['cut_offset_px']['top_bias_pct']}%")
    print("\n-- Merge habit (he leaves interior gutters inside a slice) --")
    print(f"  slices containing >=1 interior gutter: {report['merge_habit']['slices_with_interior_gutter_pct']}%")
    print(f"  slices containing >=2 interior gutters: {report['merge_habit']['slices_merged_2plus_pct']}%")
    print(f"  interior bands per 100 slices:          {report['merge_habit']['interior_bands_per_100_slices']}")
    print("\n-- Safety --")
    print(f"  slices whose cut edge touches content: {report['edge_content_pct']}%")
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()
