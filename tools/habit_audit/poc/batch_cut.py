"""
CLI Batch Cutter — ใช้ v5 (MobileNetV3 + DP count-free) ตัดภาพทั้งตอน/ทั้งเรื่อง
โดยไม่ต้องเปิด GUI

usage:
  # ตัด 1 ตอน
  python batch_cut.py --input D:\moctoonRT\M\4.สาวน้อย\1 --output D:\output\ep1

  # ตัดทั้งเรื่อง (ทุกตอน)
  python batch_cut.py --input D:\moctoonRT\M\4.สาวน้อย --output D:\output\4.สาวน้อย

  # ใช้โมเดลเฉพาะ
  python batch_cut.py --input ... --output ... --model D:\python\cutting_manga_v1\cnn_v5.pth
"""
import argparse, os, re, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# เอา repo root ขึ้น sys.path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from core.ai_predictor import AIPredictor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MODEL = str(REPO / "cnn_v5.pth")


def natural_key(text):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"([0-9]+)", text)]


def imread(path):
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


def merge_episode(ep_dir):
    """ต่อภาพตอน (กรอง blank) → PIL Image. คืน (merged_pil, n_kept, n_dropped)"""
    files = sorted([p for p in Path(ep_dir).iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS],
                   key=lambda p: natural_key(p.name))
    imgs = []
    dropped = 0
    for p in files:
        im = imread(p)
        if im is None:
            dropped += 1
            continue
        if is_blank(im):
            dropped += 1
            continue
        imgs.append(im)
    if len(imgs) < 2:
        return None, len(imgs), dropped
    w = min(im.shape[1] for im in imgs)
    for i, im in enumerate(imgs):
        if im.shape[1] != w:
            imgs[i] = cv2.resize(im, (w, im.shape[0]))
    strip = np.vstack(imgs)
    return Image.fromarray(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)), len(imgs), dropped


def is_episode_dir(p):
    """มีไฟล์ภาพอย่างน้อย 2 ไฟล์โดยตรง."""
    return Path(p).is_dir() and p.isdigit()


def cut_episode(ep_dir, predictor, output_dir, ep_name):
    """ตัด 1 ตอน → บันทึกชิ้นแพเนลลง output_dir"""
    merged, kept, dropped = merge_episode(ep_dir)
    if merged is None:
        print(f"  [{ep_name}] skip — ภาพน้อยเกินไป (kept={kept})")
        return False
    t0 = time.time()
    ok, cuts, msg = predictor.predict_cut_points(merged)
    dt = time.time() - t0
    if not ok:
        print(f"  [{ep_name}] FAIL {msg}")
        return False

    # บันทึกชิ้น
    cuts = [-1] + sorted(cuts) + [merged.height]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    for i in range(len(cuts) - 1):
        top = cuts[i] + 1
        bottom = cuts[i + 1]
        if bottom - top < 20:
            continue
        piece = merged.crop((0, top, merged.width, bottom))
        out_name = f"{ep_name}_{i + 1:03d}.jpg"
        piece.save(out_dir / out_name, "JPEG", quality=92)
        n_saved += 1
    print(f"  [{ep_name}] kept={kept} drop={dropped} cuts={len(cuts)-2} saved={n_saved} ({dt:.1f}s)")
    return True


def main():
    ap = argparse.ArgumentParser(description="v5 Batch Cutter")
    ap.add_argument("--input", required=True, help="episode dir หรือ series dir")
    ap.add_argument("--output", required=True, help="output dir")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"path โมเดล (default: {DEFAULT_MODEL})")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"ไม่พบโมเดล: {args.model}")
        sys.exit(1)

    print(f"loading model: {args.model}")
    p = AIPredictor(args.model)
    ok, msg = p.load_model()
    if not ok:
        print(f"load model FAIL: {msg}")
        sys.exit(2)
    print(f"model: {p.describe()}  device={p.device}  smooth={p.smooth} a_den={p.a_den} lam={p.lam}")
    print()

    inp = Path(args.input)
    success = 0
    total = 0

    # auto-detect: ตัวเลข = single episode; มี subdir ตัวเลข = series
    if inp.name.isdigit():
        # single episode
        total = 1
        ok = cut_episode(inp, p, args.output, inp.name)
        if ok:
            success += 1
    else:
        # series: walk numbered subdirs
        eps = sorted([d for d in inp.iterdir() if d.name.isdigit() and d.is_dir()],
                     key=lambda d: int(d.name))
        if not eps:
            print(f"ไม่พบตอน (subdir ตัวเลข) ใน {inp}")
            sys.exit(3)
        total = len(eps)
        for ep_dir in eps:
            ok = cut_episode(ep_dir, p, args.output, ep_dir.name)
            if ok:
                success += 1

    print(f"\nDone: {success}/{total} episodes  model={p.describe()}")


if __name__ == "__main__":
    main()
