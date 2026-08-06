import gc, json, re, cv2, torch, numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier

TOL_N, TOL_H = 100, 200
SERIES_ROOT = Path(r"D:\python\YT-guide\project-data\series")
FEAT_DIR = Path(__file__).resolve().parents[3] / "tools" / "habit_audit" / "poc" / "features"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def natural_key(text):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"([0-9]+)", text)]


def extract_episode(ep_dir, dino, max_slices=30):
    files = sorted([p for p in ep_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                   key=lambda p: natural_key(p.name))[:max_slices]
    imgs, hs = [], []
    for p in files:
        d = np.fromfile(str(p), np.uint8)
        im = cv2.imdecode(d, cv2.IMREAD_COLOR)
        if im is None:
            continue
        hs.append(im.shape[0])
        imgs.append(im)
    w = min(im.shape[1] for im in imgs)
    for i, im in enumerate(imgs):
        if im.shape[1] != w:
            imgs[i] = cv2.resize(im, (w, im.shape[0]))
    strip = np.vstack(imgs)
    H = strip.shape[0]
    seams = np.cumsum(hs, dtype=np.int32)[:-1]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    seg = np.array([bool(gray[max(0, s - 2):s + 3].std(axis=0).mean() < 12 or
                         (gray[max(0, s - 2):s + 3] >= 230).mean() > 0.9) for s in seams])
    feats = np.zeros((H, 384), dtype=np.float32)
    bs = 20
    DEV = next(dino.parameters()).device
    for bi in range(0, H, bs):
        batch_r = list(range(bi, min(bi + bs, H)))
        patches = []
        for r in batch_r:
            top = max(0, r - 112)
            bot = min(H, top + 224)
            crop = strip[top:bot]
            crop = cv2.resize(crop, (224, 224))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
            t = (t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            patches.append(t)
        batch = torch.stack(patches).to(DEV)
        feats[bi:bi + len(batch_r)] = dino(batch).detach().cpu().numpy()
    torch.cuda.empty_cache()
    return feats, seams, seg, H


def main():
    DEV = torch.device("cuda")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval().to(DEV)
    meta = json.load(open(FEAT_DIR / "meta.json", encoding="utf-8"))
    train_series = ["1", "be-the-actor", "black-corporation-joseon", "evolution-from-the-big-tree"]
    test_series = "i-will-raise-this-family-to-greatness"

    print("Extracting DINOv2 train features (2 episodes each, 25 slices)...")
    X_list, y_list = [], []
    for s_name in train_series:
        count = 0
        for e in meta:
            if e["series"] != s_name:
                continue
            ep_dir = SERIES_ROOT / e["series"] / "episodes" / e["episode"] / "images"
            if not ep_dir.is_dir():
                continue
            feats, seams, seg, H = extract_episode(ep_dir, dino, 25)
            lbl = np.zeros(H, dtype=np.int8)
            for s in seams:
                lbl[max(0, s - 2):min(H, s + 3)] = 1
            pos_idx = np.where(lbl > 0)[0]
            neg_idx = np.where(lbl == 0)[0]
            if len(neg_idx) > len(pos_idx) * 4:
                neg_idx = np.random.default_rng(0).choice(neg_idx, len(pos_idx) * 4, replace=False)
            idx = np.concatenate([pos_idx, neg_idx])
            X_list.append(feats[idx])
            y_list.append(lbl[idx])
            count += 1
            gc.collect()
            if count >= 2:
                break

    X_train = np.vstack(X_list).astype(np.float32)
    y_train = np.concatenate(y_list).astype(np.int8)
    print(f"train rows={len(y_train)} pos={int(y_train.sum())}")

    spacings = []
    for e in meta:
        if e["series"] not in train_series:
            continue
        safe = re.sub(r'[\\/:*?"<>|]', "_", f"{e['series']}__{e['episode']}")
        p = FEAT_DIR / f"{safe}.npz"
        if p.exists():
            z = np.load(p)
            spacings.extend(np.diff(np.concatenate([[0], z["seams"]])).tolist())
            z.close()
    spacings = np.array(spacings)
    h = {"min_seg": float(np.percentile(spacings, 15)), "max_seg": float(np.percentile(spacings, 90)),
         "median": float(np.median(spacings))}
    print(f"habit(min={h['min_seg']:.0f},max={h['max_seg']:.0f})")

    gbm = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0,
                                         early_stopping=True, validation_fraction=0.1).fit(X_train, y_train)
    gc.collect()

    print("Testing on i-will-raise (2 episodes, 30 slices)...")
    tot_hit, tot_real, n_ep = 0, 0, 0
    for e in meta:
        if e["series"] != test_series:
            continue
        ep_dir = SERIES_ROOT / e["series"] / "episodes" / e["episode"] / "images"
        if not ep_dir.is_dir():
            continue
        feats, seams, seg, H = extract_episode(ep_dir, dino, 30)
        probs = gbm.predict_proba(feats)[:, 1]
        score = np.convolve(probs, np.ones(31) / 31, mode="same")
        dist = max(1, int(h["min_seg"]) // 2)
        peaks, _ = find_peaks(score, height=0.25, distance=dist)
        hit = 0
        used = set()
        for i, s in enumerate(seams):
            tol = TOL_N if seg[i] else TOL_H
            d = np.abs(peaks - s)
            if len(d) == 0:
                continue
            best = int(np.argmin(d))
            if d[best] <= tol and best not in used:
                hit += 1
                used.add(best)
        tot_hit += hit
        tot_real += len(seams)
        n_ep += 1
        gc.collect()
        if n_ep >= 2:
            break

    print(f"\nDINOv2 recall ({n_ep} ep): {tot_hit/max(1,tot_real)*100:.1f}% ({tot_hit}/{tot_real})")
    print("(handcrafted features had 16.4% on this fold)")


if __name__ == "__main__":
    main()

