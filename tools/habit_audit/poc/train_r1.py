import argparse, gc, json, random, re, sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT / "tools" / "habit_audit" / "poc" / "split.json"
FEAT_DIR = ROOT / "tools" / "habit_audit" / "poc" / "features"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
WIN_H = 512
IN_SIZE = 512
SEED = 7


def natural_key(text):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"([0-9]+)", text)]


def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is not None and len(im.shape) == 3 and im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    return im


class BoundaryDataset(Dataset):
    def __init__(self, eps, neg_ratio=1.5, seed=SEED):
        self.refs = []
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IN_SIZE, IN_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        for e in eps:
            img_dir = Path(e["images"])
            files = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                           key=lambda p: natural_key(p.name))
            if len(files) < 2:
                continue
            w = None
            for f in files[:3]:
                im = imread(f)
                if im is not None:
                    w = im.shape[1]
                    break
            if w is None:
                continue
            for i in range(len(files) - 1):
                self.refs.append(("pos", (files[i], files[i + 1], w), 1))
            im = None
            for i in range(len(files)):
                im = imread(files[i])
                if im is not None and im.shape[1] != w:
                    im = cv2.resize(im, (w, im.shape[0]))
                if im is None or im.shape[0] < IN_SIZE + 50:
                    continue
                max_neg = max(1, int(neg_ratio * (len(files) - 1) / max(1, len(files))))
                for _ in range(max_neg):
                    margin = IN_SIZE // 4
                    y = random.randint(margin, max(margin + 1, im.shape[0] - IN_SIZE - margin))
                    self.refs.append(("neg", (files[i], y, w), 0))
        random.seed(seed)
        random.shuffle(self.refs)

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, idx):
        typ, ref, label = self.refs[idx]
        if typ == "pos":
            a = imread(ref[0])
            b = imread(ref[1])
            w = ref[2]
            if a is None or b is None:
                a = b = np.zeros((WIN_H, w, 3), dtype=np.uint8)
            if a.shape[1] != w:
                a = cv2.resize(a, (w, a.shape[0]))
            if b.shape[1] != w:
                b = cv2.resize(b, (w, b.shape[0]))
            up = a[-WIN_H // 2:]
            down = b[:WIN_H // 2]
            if up.shape[0] + down.shape[0] < WIN_H:
                pad = WIN_H - up.shape[0] - down.shape[0]
                down = np.vstack([down, np.zeros((pad, w, 3), dtype=np.uint8)]) if pad > 0 else down
            img = np.vstack([up, down])[:WIN_H]
        else:
            f, y0, w = ref
            a = imread(f)
            if a is None:
                a = np.zeros((IN_SIZE + 50, w, 3), dtype=np.uint8)
            if a.shape[1] != w:
                a = cv2.resize(a, (w, a.shape[0]))
            y0 = min(y0, max(0, a.shape[0] - IN_SIZE))
            img = a[y0:y0 + IN_SIZE]
        img = cv2.resize(img, (IN_SIZE, IN_SIZE))
        t = self.transform(img)
        return t, torch.tensor(label, dtype=torch.long)


def train_cnn(train_eps, out_path, epochs, batch, lr, workers, seed=SEED):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CNN train: {len(train_eps)} episodes | device={device} | seed={seed}")
    ds = BoundaryDataset(train_eps, seed=seed)
    print(f"samples={len(ds)}")
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=True, persistent_workers=workers > 0)
    mdl = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    mdl.classifier[3] = nn.Linear(mdl.classifier[3].in_features, 2)
    mdl.to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(mdl.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")
    for epoch in range(epochs):
        mdl.train()
        loss_sum, cor, tot = 0.0, 0, 0
        for x, y in tqdm(dl, desc=f"E{epoch+1}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda" if device.type == "cuda" else "cpu"):
                out = mdl(x)
                loss = crit(out, y)
            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            loss_sum += loss.item()
            cor += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
        acc = cor / max(1, tot) * 100
        print(f"E{epoch+1} | loss={loss_sum/max(1,len(dl)):.3f} acc={acc:.1f}%")
        torch.save(mdl.state_dict(), out_path)
    print("CNN saved:", out_path)


def train_gbm(train_eps, out_path, max_per_ep=1500):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    NORM_COLS = [0, 1, 2, 5]
    X, y = [], []
    for e in train_eps:
        safe = re.sub(r'[\\/:*?"<>|]', "_", f"{e['series']}__{e['episode']}")
        p = FEAT_DIR / f"{safe}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        feats = z["feats"].astype(np.float32)
        f = feats.copy()
        for c in NORM_COLS:
            mu, sd = f[:, c].mean(), f[:, c].std() + 1e-6
            f[:, c] = (f[:, c] - mu) / sd
        idx = np.concatenate([z["pos"], z["neg"]])
        lbl = np.concatenate([np.ones(len(z["pos"])), np.zeros(len(z["neg"]))])
        if len(idx) > max_per_ep:
            keep = np.random.default_rng(0).choice(len(idx), max_per_ep, replace=False)
            idx, lbl = idx[keep], lbl[keep]
        X.append(f[idx])
        y.append(lbl)
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int8)
    print(f"GBM train rows={len(y)} pos={int(y.sum())}")
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0,
                                         early_stopping=True, validation_fraction=0.1)
    clf.fit(X, y)
    joblib.dump(clf, out_path)
    print("GBM saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train-gbm", action="store_true")
    ap.add_argument("--train-cnn", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gbm-out", type=Path, default=ROOT / "tools" / "habit_audit" / "poc" / "gbm_r1.joblib")
    ap.add_argument("--cnn-out", type=Path, default=ROOT / "tools" / "habit_audit" / "poc" / "cnn_r1.pth")
    args = ap.parse_args()

    split = json.load(open(SPLIT, encoding="utf-8"))
    train_eps = split["train"]
    if args.train_gbm:
        train_gbm(train_eps, args.gbm_out)
    if args.train_cnn:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_cnn(train_eps, args.cnn_out, args.epochs, args.batch, args.lr, args.workers, seed=args.seed)


if __name__ == "__main__":
    main()

