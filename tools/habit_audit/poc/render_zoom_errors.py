import json, sys, re
from pathlib import Path
import numpy as np
import cv2
import eval_dp2 as D

SW, A_DEN, MD, RR = 15, 1.0, 1500, 10
CACHE = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r4")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def safe(name):
    return (name.replace("\\", "_").replace("/", "_").replace(":", "_")
            .replace("*", "_").replace("?", "_").replace('"', "_")
            .replace("<", "_").replace(">", "_").replace("|", "_"))

def natural_key(t):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"([0-9]+)", t)]

def imread_color(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def select_from_cache(z):
    sv = np.convolve(z["sv"], np.ones(SW) / SW, mode="same")
    score = sv - A_DEN * (z["density"][z["yc"]] / max(1.0, float(z["density"].max())))
    cuts = D.dp_select(z["yc"], score, len(z["seams"]), MD)
    den = z["density"]
    out = []
    for c in cuts:
        lo = max(0, c - RR); hi = min(int(z["H"]) - 1, c + RR)
        out.append(lo + int(np.argmin(den[lo:hi + 1])))
    return np.array(out, dtype=np.int32)

def matched_status(cuts, seams):
    """Per real seam: distance to the NEAREST AI line (intuitive 'how close AI got')."""
    cuts = np.sort(cuts)
    dists = []
    for s in seams:
        dists.append(int(np.abs(cuts - s).min()) if len(cuts) else -1)
    dists = np.array(dists)
    status = dists <= 50
    return status, dists

def render_zoom(series, episode, out_path, nseams=5):
    name = safe(f"{series}__{episode}")
    z = np.load(CACHE / f"{name}.npz")
    seams = np.sort(z["seams"])
    seg = z["seg"]
    cuts = select_from_cache(z)
    status, dists = matched_status(cuts, seams)

    # reload strip
    ep_dir = Path(r"D:\python\YT-guide\project-data\series") / series / "episodes" / episode / "images"
    files = sorted([p for p in ep_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                   key=lambda p: natural_key(p.name))
    imgs = [imread_color(p) for p in files]
    if not imgs:
        print("no images")
        return
    w = min(im.shape[1] for im in imgs)
    for i, im in enumerate(imgs):
        if im.shape[1] != w:
            imgs[i] = cv2.resize(im, (w, im.shape[0]))
    strip = np.vstack(imgs)
    H = strip.shape[0]
    hs = [im.shape[0] for im in imgs]

    # pick a section with a mix of matched/unmatched
    n = len(seams)
    best_i0 = 0
    for i0 in range(0, max(1, n - nseams)):
        window = status[i0:i0 + nseams]
        if window.any() and (~window).any():
            best_i0 = i0
            break
    i1 = min(n, best_i0 + nseams)
    y0 = max(0, int(seams[best_i0]) - 500)
    y1 = min(H, int(seams[i1 - 1]) + 500)
    sec = strip[y0:y1]
    SH = sec.shape[0]
    ZW = 800
    zscale = ZW / sec.shape[1]
    ZH = int(SH * zscale)

    # tint slices
    img = sec.copy().astype(np.float32)
    for si in range(len(imgs)):
        a = 0 if si == 0 else seams[si - 1]
        b = H if si == len(imgs) - 1 else seams[si]
        a, b = max(a, y0), min(b, y1)
        if a >= b:
            continue
        col = (255, 190, 130) if si % 2 == 0 else (130, 175, 255)
        img[a - y0:b - y0, :, :] = img[a - y0:b - y0, :, :] * 0.8 + np.array(col, np.float32) * 0.2
    img = img.astype(np.uint8)
    img = cv2.resize(img, (ZW, ZH))

    # score panel
    panel = np.zeros((ZH, 160, 3), dtype=np.uint8)
    panel[:] = (25, 25, 25)
    yc = z["yc"]; sv = np.convolve(z["sv"], np.ones(SW) / SW, mode="same")
    m = (yc >= y0) & (yc < y1)
    py, pv = yc[m], sv[m]
    if len(py):
        pts = np.vstack([(py - y0) * zscale,
                         (np.clip(pv, 0, 1) * 140 + 10).astype(np.int32)]).T.astype(np.int32)
        pts[:, 1] = np.clip(pts[:, 1], 0, 158)
        cv2.polylines(panel, [pts], False, (255, 0, 255), 1)
    for t in (0.5, 0.8):
        x = int(t * 140 + 10)
        cv2.line(panel, (x, 0), (x, ZH), (60, 60, 60), 1)

    # draw lines + labels
    label_x = 10
    for i in range(best_i0, i1):
        s = int(seams[i])
        yy = int((s - y0) * zscale)
        cv2.line(img, (0, yy), (ZW, yy), (0, 255, 0), 2)
        cv2.rectangle(img, (0, max(0, int((s - y0 - 50) * zscale))),
                      (ZW, min(ZH, int((s - y0 + 50) * zscale))), (0, 255, 255), 1)
        ok = status[i]
        d = dists[i] if dists[i] is not None else -1
        color = (0, 255, 0) if ok else (0, 0, 255)
        txt = f"real {i+1} | AI {d}px" + (" OK" if ok else " MISS")
        cv2.putText(img, txt, (label_x, yy - 6 if yy > 30 else yy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        cv2.putText(img, txt, (label_x, yy - 6 if yy > 30 else yy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    for c in cuts:
        if y0 <= c < y1:
            yy = int((c - y0) * zscale)
            cv2.line(img, (0, yy), (ZW, yy), (0, 0, 255), 1)
    # AI ticks on panel (red), seam ticks green
    for i in range(best_i0, i1):
        yy = int((seams[i] - y0) * zscale)
        cv2.line(panel, (0, yy), (159, yy), (0, 255, 0), 1)
    for c in cuts:
        if y0 <= c < y1:
            yy = int((c - y0) * zscale)
            cv2.line(panel, (0, yy), (159, yy), (0, 0, 255), 1)

    # header
    n_ok = D.match_1to1(cuts, seams)   # strict 1:1 (consistent with eval)
    bar = np.zeros((70, ZW + 160, 3), dtype=np.uint8)
    lines = [
        f"  {series} / {episode}   |   real seams = {len(seams)}   AI cuts = {len(cuts)}",
        f"  matched +-50px (1:1) = {n_ok}/{len(seams)} = {n_ok/len(seams)*100:.0f}%   |   "
        f"green=real seam  yellow=+-50 band  red=AI cut  magenta=AI score",
    ]
    for li, t in enumerate(lines):
        cv2.putText(bar, t, (8, 24 + li * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    canvas = np.vstack([bar, np.hstack([img, panel])])
    canvas = cv2.resize(canvas, (int(canvas.shape[1] * 0.55), int(canvas.shape[0] * 0.55)))
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"saved {out_path} {canvas.shape[1]}x{canvas.shape[0]} | matched {n_ok}/{len(seams)}")

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="zoom_errors.jpg")
args = ap.parse_args()

# pick 3 episodes: good / mid / hard
render_zoom("1", "ตอนที่ 150", Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\zoom_good.jpg"))
render_zoom("evolution-from-the-big-tree", "ตอนที่ 201", Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\zoom_mid.jpg"))
render_zoom("i-will-raise-this-family-to-greatness", "ตอนที่ 26", Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\zoom_hard.jpg"))
