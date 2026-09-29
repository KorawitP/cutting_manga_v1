import argparse, gc, json, re, time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier
import torchvision.models as models
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT / "tools" / "habit_audit" / "poc" / "split.json"
FEAT_DIR = ROOT / "tools" / "habit_audit" / "poc" / "features"
NORM_COLS = [0, 1, 2, 5]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TOL_N, TOL_H = 100, 200
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRIDE = 16
WIN_H = 512

import sys
sys.path.insert(0, str(ROOT / "tools" / "habit_audit" / "poc"))


def natural_key(text):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"([0-9]+)", text)]


def imread_color(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return im


def norm_feats(f):
    out = f.astype(np.float32).copy()
    for c in NORM_COLS:
        mu, sd = f[:, c].mean(), f[:, c].std() + 1e-6
        out[:, c] = (f[:, c] - mu) / sd
    return out


def build_features_for_episode(ep_dir):
    """merge slices -> per-row features + seams + seg. Returns dict or None."""
    files = sorted([p for p in ep_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                   key=lambda p: natural_key(p.name))
    if len(files) < 2:
        return None
    imgs, hs = [], []
    for p in files:
        im = imread_color(p)
        if im is None:
            continue
        hs.append(im.shape[0])
        imgs.append(im)
    if not imgs:
        return None
    w = min(im.shape[1] for im in imgs)
    for i, im in enumerate(imgs):
        if im.shape[1] != w:
            imgs[i] = cv2.resize(im, (w, im.shape[0]))
    strip = np.vstack(imgs)
    H = strip.shape[0]
    seams = np.cumsum(hs, dtype=np.int32)[:-1]

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    sobel = cv2.addWeighted(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), 1,
                            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3), 1, 0)
    sobel = cv2.convertScaleAbs(sobel).astype(np.float32)
    std = gray.std(axis=1).astype(np.float32)
    white = (gray >= 230).mean(axis=1).astype(np.float32)
    edge = sobel.mean(axis=1).astype(np.float32)
    is_g = ((std < 12) | (white >= 0.97)) & (edge < 4)
    gw = np.zeros(H, dtype=np.int32)
    i = 0
    while i < H:
        if is_g[i]:
            j = i
            while j < H and is_g[j]:
                j += 1
            gw[i:j] = j - i
            i = j
        else:
            i += 1
    density = np.convolve(edge, np.ones(41, dtype=np.float32) / 41, mode="same")
    dist_top = np.arange(H, dtype=np.float32) / max(1, H - 1)
    feats = np.stack([std, white, edge, is_g.astype(np.float32), gw.astype(np.float32),
                      density, dist_top], axis=1)
    seg = np.array([bool(is_g[max(0, s - 2):s + 3].mean() >= 0.6) for s in seams])
    return {"strip": strip, "feats": feats, "seams": seams, "seg": seg, "density": density}


def refine_peaks(peaks, density, H, r=20):
    out = []
    for pk in peaks:
        lo = max(0, int(pk) - r)
        hi = min(H - 1, int(pk) + r)
        out.append(lo + int(np.argmin(density[lo:hi + 1])))
    return np.array(out, dtype=np.int32) if out else np.array([], dtype=np.int32)


def run_gbm(ep, gbm):
    feats = norm_feats(ep["feats"])
    probs = gbm.predict_proba(feats)[:, 1]
    score = probs.copy()
    gw = feats[:, 4]
    score[gw < 5] *= 0.3
    score = np.convolve(score, np.ones(31) / 31, mode="same")
    dist = max(1, int(30 // 2))
    pk, _ = find_peaks(score, height=0.25, distance=dist)
    return refine_peaks(pk, ep["density"], ep["strip"].shape[0])


def run_cnn(ep, model):
    strip = ep["strip"]
    H = strip.shape[0]
    strip_small = cv2.resize(strip, (512, H))
    strip_rgb = cv2.cvtColor(strip_small, cv2.COLOR_BGR2RGB)  # (H,512,3) uint8
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
                    probs = torch.softmax(model(bb), dim=1)[:, 1].cpu().numpy()
                for i, p in enumerate(probs):
                    y_centers.append(int(batch_y[i] + WIN_H // 2))
                    score_vals.append(float(p))
                batch_t, batch_y = [], []
        if batch_t:
            bb = torch.stack(batch_t)
            with torch.no_grad():
                probs = torch.softmax(model(bb), dim=1)[:, 1].cpu().numpy()
            for i, p in enumerate(probs):
                y_centers.append(int(batch_y[i] + WIN_H // 2))
                score_vals.append(float(p))
        del t_chunk
        torch.cuda.empty_cache()
    yc = np.array(y_centers, dtype=np.int32)
    sv = np.array(score_vals, dtype=np.float32)
    sv_sm = np.convolve(sv, np.ones(31) / 31, mode="same")
    dist_samples = max(1, 30 // STRIDE // 2)
    pk, _ = find_peaks(sv_sm, height=0.3, distance=dist_samples)
    peaks = yc[pk]
    return refine_peaks(peaks, ep["density"], H)


def eval_episode(pred, ep):
    seams, seg = ep["seams"], ep["seg"]
    if len(seams) == 0:
        return None
    hit = 0
    used = set()
    for i, s in enumerate(seams):
        tol = TOL_N if seg[i] else TOL_H
        d = np.abs(pred - s)
        if len(d) == 0:
            continue
        best = int(np.argmin(d))
        if d[best] <= tol and best not in used:
            hit += 1
            used.add(best)
    recall = hit / len(seams)
    if len(pred):
        matched = sum(1 for pc in pred if np.abs(seams - pc).min() <= max(TOL_N, TOL_H))
        precision = matched / len(pred)
    else:
        precision = 0.0
    return recall, precision, len(pred), len(seams)


def load_gbm(model_path):
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0,
                                          early_stopping=True, validation_fraction=0.1)


def load_cnn(model_path):
    mdl = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    mdl.classifier[3] = nn.Linear(mdl.classifier[3].in_features, 2)
    mdl.load_state_dict(torch.load(model_path, map_location=DEV))
    mdl.eval().to(DEV)
    return mdl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbm", type=Path, required=True)
    ap.add_argument("--cnn", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=SPLIT)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    split = json.load(open(args.split, encoding="utf-8"))
    val_eps = split["val"]
    print(f"Evaluating on {len(val_eps)} held-out episodes")

    import joblib
    gbm = joblib.load(args.gbm)
    cnn = load_cnn(args.cnn)

    results = {"gbm": {}, "cnn": {}}
    tot = {"gbm": {"hit": 0, "real": 0, "prec": 0, "pred": 0, "t": 0.0},
           "cnn": {"hit": 0, "real": 0, "prec": 0, "pred": 0, "t": 0.0}}
    for e in val_eps:
        ep_dir = Path(e["images"])
        ep = build_features_for_episode(ep_dir)
        if ep is None:
            continue
        # GBM
        t0 = time.time()
        pred = run_gbm(ep, gbm)
        dt = time.time() - t0
        r = eval_episode(pred, ep)
        if r:
            rec, prec, np_, ns = r
            tot["gbm"]["hit"] += rec * ns
            tot["gbm"]["real"] += ns
            tot["gbm"]["prec"] += prec * np_
            tot["gbm"]["pred"] += np_
            tot["gbm"]["t"] += dt
        # CNN
        t0 = time.time()
        pred = run_cnn(ep, cnn)
        dt = time.time() - t0
        r = eval_episode(pred, ep)
        if r:
            rec, prec, np_, ns = r
            tot["cnn"]["hit"] += rec * ns
            tot["cnn"]["real"] += ns
            tot["cnn"]["prec"] += prec * np_
            tot["cnn"]["pred"] += np_
            tot["cnn"]["t"] += dt
        gc.collect()

    summary = {}
    for m in ("gbm", "cnn"):
        t = tot[m]
        summary[m] = {
            "recall": t["hit"] / max(1, t["real"]) * 100,
            "precision": t["prec"] / max(1, t["pred"]) * 100,
            "pred_cuts": t["pred"],
            "real_cuts": t["real"],
            "total_time_s": round(t["t"], 1),
            "per_episode_s": round(t["t"] / max(1, len(val_eps)), 2),
        }
    print("\n==== HELD-OUT EVAL (42 episodes) ====")
    for m in ("gbm", "cnn"):
        s = summary[m]
        print(f"  {m:4s} recall={s['recall']:.1f}% precision={s['precision']:.1f}% "
              f"({s['pred_cuts']}/{s['real_cuts']}) | {s['total_time_s']}s total "
              f"({s['per_episode_s']}s/ep)")
    delta = summary["cnn"]["recall"] - summary["gbm"]["recall"]
    print(f"  delta (cnn-gbm) = {delta:+.1f}%")
    print(f"  CNN speed = {summary['gbm']['per_episode_s'] / max(1, summary['cnn']['per_episode_s']):.1f}x slower than GBM")

    out = args.out or (args.gbm.parent / f"round_{args.gbm.parent.name}_eval.json")
    out = args.gbm.parent / "eval_round.json"
    json.dump(summary, open(out, "w"), indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
