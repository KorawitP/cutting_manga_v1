import json, re, sys, time, gc
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models

ROOT = Path(__file__).resolve().parents[3]
POC = ROOT / "tools" / "habit_audit" / "poc"
SPLIT = json.load(open(POC / "split_v3.json", encoding="utf-8"))
CNN = POC / "cnn_r4.pth"
OUT = POC / "cnn_cache_r4_v3"
OUT.mkdir(exist_ok=True)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIN_H = 512
STRIDE = 16

def safe(name):
    return (name.replace("\\", "_").replace("/", "_").replace(":", "_")
            .replace("*", "_").replace("?", "_").replace('"', "_")
            .replace("<", "_").replace(">", "_").replace("|", "_"))

def natural_key(text):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"([0-9]+)", text)]

def imread_color(path):
    d = np.fromfile(str(path), dtype=np.uint8)
    if len(d) == 0:
        return None
    return cv2.imdecode(d, cv2.IMREAD_COLOR)

def is_blank(im):
    small = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)
    arr = small.reshape(-1, 3)
    uniq = len(np.unique(arr, axis=0))
    if uniq <= 2:
        return True
    if uniq <= 6 and (arr.mean() > 245 or arr.mean() < 15):
        return True
    return False

def build_strip(edir):
    all_files = sorted([p for p in Path(edir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                       key=lambda p: natural_key(p.name))
    imgs, hs = [], []
    w = None
    for p in all_files:
        im = imread_color(p)
        if im is None or is_blank(im):
            continue
        hs.append(im.shape[0])
        imgs.append(im)
        if w is None:
            w = im.shape[1]
    if len(imgs) < 2:
        return None
    for i, im in enumerate(imgs):
        if im.shape[1] != w:
            imgs[i] = cv2.resize(im, (w, im.shape[0]))
    strip = np.vstack(imgs)
    seams = np.cumsum(hs, dtype=np.int32)[:-1]
    return strip, seams, w

def load_cnn():
    mdl = models.mobilenet_v3_small(weights=None)
    mdl.classifier[3] = nn.Linear(mdl.classifier[3].in_features, 2)
    mdl.load_state_dict(torch.load(CNN, map_location=DEV))
    mdl.to(DEV).eval()
    return mdl

def run_cnn(mdl, strip):
    H = strip.shape[0]
    strip_small = cv2.resize(strip, (512, H))
    strip_rgb = cv2.cvtColor(strip_small, cv2.COLOR_BGR2RGB)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEV)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEV)
    y_centers, score_vals = [], []
    bs = 32
    CHUNK = 8192
    for chunk_start in range(0, H - WIN_H, CHUNK):
        chunk_end = min(H, chunk_start + CHUNK + WIN_H)
        arr = strip_rgb[chunk_start:chunk_end]
        t_chunk = torch.from_numpy(arr).permute(2, 0, 1).float().to(DEV) / 255.0
        t_chunk = (t_chunk - mean) / std
        batch_t, batch_y = [], []
        for sy in range(chunk_start, min(chunk_start + CHUNK, H - WIN_H, chunk_end - WIN_H), STRIDE):
            local = sy - chunk_start
            batch_t.append(t_chunk[:, local:local + WIN_H, :])
            batch_y.append(sy)
            if len(batch_t) == bs:
                bb = torch.stack(batch_t)
                with torch.no_grad():
                    probs = torch.softmax(mdl(bb), dim=1)[:, 1].cpu().numpy()
                for i, p in enumerate(probs):
                    y_centers.append(int(batch_y[i] + WIN_H // 2))
                    score_vals.append(float(p))
                batch_t, batch_y = [], []
        if batch_t:
            bb = torch.stack(batch_t)
            with torch.no_grad():
                probs = torch.softmax(mdl(bb), dim=1)[:, 1].cpu().numpy()
            for i, p in enumerate(probs):
                y_centers.append(int(batch_y[i] + WIN_H // 2))
                score_vals.append(float(p))
        del t_chunk
        torch.cuda.empty_cache()
    return np.array(y_centers, dtype=np.int32), np.array(score_vals, dtype=np.float32)

mdl = load_cnn()
print(f"model loaded from {CNN}, device={DEV}", flush=True)
t0 = time.time()
done = 0
for e in SPLIT["val"]:
    name = safe(f"{e['series']}__{e['episode']}")
    np_path = OUT / f"{name}.npz"
    if np_path.exists():
        continue
    r = build_strip(e["images"])
    if r is None:
        print(f"  skip {name} (strip fail)", flush=True)
        continue
    strip, seams, w = r
    yc, sv = run_cnn(mdl, strip)
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    sobel = cv2.addWeighted(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), 1,
                            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3), 1, 0)
    sobel = cv2.convertScaleAbs(sobel).astype(np.float32)
    density = np.convolve(sobel.mean(axis=1), np.ones(41, dtype=np.float32) / 41, mode="same")
    np.savez_compressed(np_path, yc=yc, sv=sv, seams=seams, density=density, H=strip.shape[0])
    done += 1
    print(f"{time.time()-t0:.0f}s [{done}] {name} windows={len(yc)} seams={len(seams)}", flush=True)
    del strip, yc, sv
    gc.collect()
print(f"cache done in {time.time()-t0:.0f}s, episodes={done}")

