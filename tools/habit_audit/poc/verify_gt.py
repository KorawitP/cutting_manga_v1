import json, sys, re
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import eval_both as E

CACHE_R3 = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r3")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")

def safe(name):
    return (name.replace("\\", "_").replace("/", "_").replace(":", "_")
            .replace("*", "_").replace("?", "_").replace('"', "_")
            .replace("<", "_").replace(">", "_").replace("|", "_"))

# pick a val episode that has a hard-case seam (cut on content)
split = json.load(open(SPLIT, encoding="utf-8"))
chosen = None
for e in split["val"]:
    name = safe(f"{e['series']}__{e['episode']}")
    z = np.load(CACHE_R3 / f"{name}.npz")
    if len(z["seams"]) >= 8 and (~z["seg"]).any():
        chosen = (e, name, z)
        break
if chosen is None:
    e, name, z = split["val"][0], safe(f"{split['val'][0]['series']}__{split['val'][0]['episode']}"), None
    z = np.load(CACHE_R3 / f"{name}.npz")
    chosen = (e, name, z)
e, name, z = chosen
print(f"episode: {e['series']} / {e['episode']}")

# ---- reload full strip for rendering ----
ep_dir = Path(e["images"])
files = sorted([p for p in ep_dir.iterdir() if p.is_file() and p.suffix.lower() in E.IMAGE_EXTS],
               key=lambda p: E.natural_key(p.name))
imgs = [E.imread_color(p) for p in files]
w = min(im.shape[1] for im in imgs)
for i, im in enumerate(imgs):
    if im.shape[1] != w:
        imgs[i] = cv2.resize(im, (w, im.shape[0]))
strip = np.vstack(imgs)
H = strip.shape[0]
hs = [im.shape[0] for im in imgs]
seams = np.cumsum(hs, dtype=np.int32)[:-1]
seg = z["seg"]
print(f"strip {strip.shape[1]}x{strip.shape[0]} | seams={len(seams)} "
      f"hard={(~seg).sum()} gutter={seg.sum()}")

# distance from each seam to nearest CNN peak
yc = z["yc"]; sv = np.convolve(z["sv"], np.ones(11) / 11, mode="same")
pk, _ = E.find_peaks(sv, distance=max(1, 30 // 16 // 2))
pk_y = yc[pk]
dists = np.abs(seams[:, None] - pk_y[None, :]).min(axis=1)
for t in (30, 50, 100, 200, 300):
    print(f"  seams with CNN peak within +-{t}px: {(dists<=t).mean()*100:.1f}%")

# ---------- OVERVIEW ----------
OW = 380
scale = OW / strip.shape[1]
scale = min(scale, 20000 / H)   # cap output height (JPEG max 65500px)
OH = int(H * scale)
ov = cv2.resize(strip, (OW, OH))
ov = ov.astype(np.float32)
for i, s in enumerate(seams):
    y = int(s * scale)
    cv2.rectangle(ov, (0, max(0, int((s - 50) * scale))), (OW, min(OH, int((s + 50) * scale))),
                  (0, 255, 255), -1)
cv2.addWeighted(ov, 0.55, cv2.resize(strip, (OW, OH)).astype(np.float32), 0.45, 0, ov)
ov = ov.astype(np.uint8)
for s in seams:
    cv2.line(ov, (0, int(s * scale)), (OW, int(s * scale)), (0, 255, 0), 1)
bar = np.zeros((40, OW, 3), dtype=np.uint8)
cv2.putText(bar, f"OVERVIEW {e['series']} {e['episode']} | green=real seam  yellow=+-50px band",
            (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
canvas = np.vstack([bar, ov])
out1 = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc") / "verify_overview.jpg"
cv2.imwrite(str(out1), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("saved", out1, canvas.shape[1], "x", canvas.shape[0])

# ---------- ZOOM around a hard seam ----------
hard_i = int(np.where(~seg)[0][0])
center_seam = seams[hard_i]
i0 = max(0, hard_i - 2)
i1 = min(len(seams), hard_i + 3)
y0 = seams[i0] - 400
y1 = seams[i1 - 1] + 400
y0, y1 = max(0, y0), min(H, y1)
sec = strip[y0:y1]
SH = sec.shape[0]
ZW = 760
zscale = ZW / sec.shape[1]
ZH = int(SH * zscale)

# tint alternate slices + slice labels
img = sec.copy().astype(np.float32)
for si in range(len(imgs)):
    a = 0 if si == 0 else seams[si - 1]
    b = H if si == len(imgs) - 1 else seams[si]
    a, b = max(a, y0), min(b, y1)
    if a >= b:
        continue
    col = (255, 180, 120) if si % 2 == 0 else (120, 170, 255)
    img[a - y0:b - y0, :, :] = img[a - y0:b - y0, :, :] * 0.82 + np.array(col, np.float32) * 0.18
img = img.astype(np.uint8)
img = cv2.resize(img, (ZW, ZH))

# score curve panel (width 150)
panel = np.zeros((ZH, 150, 3), dtype=np.uint8)
panel[:] = (28, 28, 28)
yc_in = (yc >= y0) & (yc < y1)
py = yc[yc_in]; pv = sv[yc_in]
if len(py):
    px = (pv / pv.max() * 130 + 8).astype(np.int32)
    pts = np.vstack([(py - y0) * zscale, px]).T.astype(np.int32)
    pts[:, 1] = np.clip(pts[:, 1], 0, 148)
    cv2.polylines(panel, [pts], False, (255, 0, 255), 1)
    for thr, colr in ((0.5, (120, 120, 120)), (0.3, (70, 70, 70))):
        x = int(thr / pv.max() * 130 + 8)
        cv2.line(panel, (x, 0), (x, ZH), colr, 1)
for s in seams:
    if not (y0 <= s < y1):
        continue
    yy = int((s - y0) * zscale)
    cv2.line(img, (0, yy), (ZW, yy), (0, 255, 0), 2)          # real seam
    for off, colr in ((50, (0, 255, 255)), (100, (0, 165, 255)), (200, (0, 0, 255))):
        cv2.rectangle(img, (0, max(0, int((s - y0 - off) * zscale))),
                      (ZW, min(ZH, int((s - y0 + off) * zscale))), colr, 1)
    cv2.line(panel, (0, yy), (149, yy), (0, 255, 0), 1)
# slice index labels
for si in range(len(imgs)):
    a = 0 if si == 0 else seams[si - 1]
    b = H if si == len(imgs) - 1 else seams[si]
    if b <= y0 or a >= y1:
        continue
    mid = int(((max(a, y0) + min(b, y1)) / 2 - y0) * zscale)
    cv2.putText(img, f"S{si+1}", (8, mid), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"S{si+1}", (8, mid), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

# legend
legend = np.zeros((100, ZW + 150, 3), dtype=np.uint8)
items = [("green = real cut (file boundary)", (0, 255, 0)),
         ("yellow = +-50px", (0, 255, 255)),
         ("orange = +-100px", (0, 165, 255)),
         ("red = +-200px", (0, 0, 255)),
         ("magenta curve = CNN score", (255, 0, 255))]
for i, (t, c) in enumerate(items):
    cv2.rectangle(legend, (10, 10 + i * 20), (30, 30 + i * 20), c, -1)
    cv2.putText(legend, t, (38, 26 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
cv2.putText(legend, "alternate slice tint = where each FILE starts/ends (verify seams)",
            (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

zoom = np.hstack([img, panel])
canvas2 = np.vstack([legend, zoom])
out2 = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc") / "verify_zoom.jpg"
cv2.imwrite(str(out2), canvas2, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("saved", out2, canvas2.shape[1], "x", canvas2.shape[0])
