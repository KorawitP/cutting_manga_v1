import argparse
import gc
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SERIES_ROOT = Path(r"D:\python\YT-guide\project-data\series")
OUT_DIR = ROOT / "tools" / "habit_audit" / "poc" / "features"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

WHITE_THRESH = 230
WHITE_RATIO = 0.97
UNIFORM_STD = 12.0
EDGE_THRESH = 4.0
MIN_GUTTER_BAND = 3
SEAM_TOL = 2


def natural_key(text):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"([0-9]+)", text)]


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


def row_features_of_slice(gray):
    h = gray.shape[0]
    sobel = cv2.addWeighted(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), 1,
                            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3), 1, 0)
    sobel = cv2.convertScaleAbs(sobel).astype(np.float32)
    std = gray.std(axis=1).astype(np.float32)
    white = (gray >= WHITE_THRESH).mean(axis=1).astype(np.float32)
    edge = sobel.mean(axis=1).astype(np.float32)
    return std, white, edge


def gutter_band_widths(is_gutter):
    """per-row width of the gutter band containing it (0 if not gutter)"""
    widths = np.zeros(len(is_gutter), dtype=np.int32)
    i = 0
    n = len(is_gutter)
    while i < n:
        if is_gutter[i]:
            j = i
            while j < n and is_gutter[j]:
                j += 1
            w = j - i
            widths[i:j] = w
            i = j
        else:
            i += 1
    return widths


def process_episode(series_name, ep_name, images_dir, neg_cap, seed):
    rng = np.random.default_rng(seed + sum(ord(c) for c in ep_name))
    files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                   key=lambda p: natural_key(p.name))
    if len(files) < 3:
        return None

    std_list, white_list, edge_list = [], [], []
    heights = []
    widths = set()
    for p in files:
        g = imread_gray(p)
        if g is None:
            continue
        widths.add(g.shape[1])
        std_list.append(g.shape[0])
    if not widths:
        return None
    w = sorted(widths)[0]

    std_rows, white_rows, edge_rows = [], [], []
    heights = []
    for p in files:
        g = imread_gray(p)
        if g is None:
            continue
        if g.shape[1] != w:
            g = cv2.resize(g, (w, g.shape[0]))
        s, wh, e = row_features_of_slice(g)
        std_rows.append(s)
        white_rows.append(wh)
        edge_rows.append(e)
        heights.append(g.shape[0])
        del g
    gc.collect()

    std = np.concatenate(std_rows)
    white = np.concatenate(white_rows)
    edge = np.concatenate(edge_rows)
    del std_rows, white_rows, edge_rows
    gc.collect()

    H = len(std)
    is_gutter = ((std < UNIFORM_STD) | (white >= WHITE_RATIO)) & (edge < EDGE_THRESH)
    gwidth = gutter_band_widths(is_gutter).astype(np.float32)
    # content density = smoothed edge (±20px window)
    k = 41
    kernel = np.ones(k, dtype=np.float32) / k
    content_density = np.convolve(edge, kernel, mode="same").astype(np.float32)
    dist_top = (np.arange(H, dtype=np.float32) / max(1, H - 1))

    feats = np.stack([std, white, edge, is_gutter.astype(np.float32),
                      gwidth, content_density, dist_top], axis=1)

    cum = np.cumsum(heights)
    seams = cum[:-1].astype(np.int32)  # boundaries between files
    seam_in_gutter = np.array([bool(is_gutter[max(0, s - SEAM_TOL):s + SEAM_TOL + 1].mean() >= 0.6) for s in seams])

    # --- labeled training rows ---
    pos = []
    for s in seams:
        for dy in range(-SEAM_TOL, SEAM_TOL + 1):
            y = int(s) + dy
            if 0 <= y < H:
                pos.append(y)
    pos = np.array(sorted(set(pos)), dtype=np.int32)

    # negatives: interior gutter band centers (user did NOT cut) + random content rows
    neg = []
    for i in range(len(files) - 1):
        seg_top = int(cum[i - 1]) if i > 0 else 0
        seg_bot = int(cum[i]) if i < len(files) else H
        band_rows = np.where(is_gutter[max(0, seg_top):seg_bot])[0] + max(0, seg_top)
        # exclude near seams
        near = np.zeros(len(band_rows), dtype=bool)
        for s in seams:
            near |= np.abs(band_rows - s) <= 5
        band_rows = band_rows[~near]
        if len(band_rows):
            # center of each band
            bs = gutter_band_widths(is_gutter)
            # take band centers by picking every run middle
            centers = []
            i0 = 0
            while i0 < H:
                if is_gutter[i0]:
                    i1 = i0
                    while i1 < H and is_gutter[i1]:
                        i1 += 1
                    c = (i0 + i1) // 2
                    if seg_top <= c < seg_bot and not np.any(np.abs(np.array(seams) - c) <= 5):
                        centers.append(c)
                    i0 = i1
                else:
                    i0 += 1
            neg.extend(centers)
    # random content rows (cap)
    content_mask = ~is_gutter
    content_idx = np.where(content_mask)[0]
    near_seam = np.zeros(len(content_idx), dtype=bool)
    for s in seams:
        near_seam |= np.abs(content_idx - s) <= 5
    content_idx = content_idx[~near_seam]
    if len(content_idx) > neg_cap:
        content_idx = rng.choice(content_idx, size=neg_cap, replace=False)
    neg.extend(int(y) for y in content_idx)
    neg = np.array(sorted(set(neg)), dtype=np.int32)

    return {
        "series": series_name, "episode": ep_name,
        "seams": seams, "seam_in_gutter": seam_in_gutter,
        "H": H, "n_files": len(files),
        "pos": pos, "neg": neg,
        "feats_path": None,
    }, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-root", type=Path, default=DEFAULT_SERIES_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--neg-cap", type=int, default=800, help="max random content negatives per episode")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-series", type=int, default=0)
    ap.add_argument("--limit-episodes", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not args.series_root.exists():
        raise FileNotFoundError(args.series_root)
    args.out.mkdir(parents=True, exist_ok=True)

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
                jobs.append((series.name, ep.name, images_dir, args.neg_cap, args.seed))

    print(f"Episodes to extract: {len(jobs)} (workers={args.workers})")

    # memory-safe: use a process pool with modest workers; each worker holds 1 episode at a time
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    meta = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_episode, *j): j[:2] for j in jobs}
        for fut in tqdm(futs, total=len(jobs)):
            series, ep = futs[fut]
            try:
                res, feats = fut.result()
            except Exception as e:
                print(f"ERR {series}/{ep}: {e}")
                continue
            if res is None:
                continue
            # save features for this episode (needed for simulation)
            safe = re.sub(r'[\\/:*?"<>|]', "_", f"{series}__{ep}")
            fpath = args.out / f"{safe}.npz"
            np.savez_compressed(fpath, feats=feats, seams=res["seams"],
                                seam_in_gutter=res["seam_in_gutter"],
                                pos=res["pos"], neg=res["neg"])
            res["feats_path"] = str(fpath)
            meta.append({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in res.items() if k != "feats_path"})
            meta[-1]["feats_path"] = str(fpath)
            del res, feats
            gc.collect()

    import json
    with open(args.out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"Done. {len(meta)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
