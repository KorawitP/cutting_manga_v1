import argparse, gc, json, random, re, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (Progress, SpinnerColumn, BarColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table
from rich import box

ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT / "tools" / "habit_audit" / "poc" / "split.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
WIN_H = 512
IN_SIZE = 512
OFFSET_MAX = 48        # positive: boundary can be up to +-48px from window center
NEAR_MIN, NEAR_MAX = 60, 280   # near-miss negatives: center this far from a seam
NON_TTY = not sys.stdout.isatty()


def natural_key(text):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"([0-9]+)", text)]


def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is not None and len(im.shape) == 3 and im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    return im


class BoundaryDatasetV2(Dataset):
    """positives with offset jitter, near-miss negatives near seams, hflip aug."""

    def __init__(self, eps, neg_ratio=1.5, seed=7, near_frac=0.5):
        self.refs = []
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IN_SIZE, IN_SIZE)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        rng = random.Random(seed)
        n_ep_done = 0

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
            # heights of slices (for near-miss positioning)
            hs = []
            for f in files:
                im = imread(f)
                hs.append(im.shape[0] if im is not None else 0)
            # positive windows at each seam
            for i in range(len(files) - 1):
                self.refs.append(("pos", (files[i], files[i + 1], w), 1))
            # negatives: near-miss + mid-slice
            n_pos = max(1, len(files) - 1)
            n_neg = max(1, int(neg_ratio * n_pos))
            n_near = int(n_neg * near_frac)
            n_mid = n_neg - n_near
            # near-miss: pick slices that are tall enough, window near its bottom seam
            tall = [i for i, h in enumerate(hs) if h >= NEAR_MAX + WIN_H]
            for _ in range(n_near):
                if not tall:
                    break
                i = rng.choice(tall)
                h = hs[i]
                # center offset above bottom seam between NEAR_MIN..NEAR_MAX
                off = rng.randint(NEAR_MIN, min(NEAR_MAX, h - WIN_H))
                y0 = h - WIN_H - off  # window top; center is h-off (near seam)
                y0 = max(0, min(y0, h - WIN_H))
                self.refs.append(("slice", (files[i], y0, w), 0))
            for _ in range(n_mid):
                i = rng.randrange(len(files))
                if hs[i] < WIN_H + 100:
                    continue
                y0 = rng.randint(IN_SIZE // 4, max(IN_SIZE // 4 + 1, hs[i] - IN_SIZE - IN_SIZE // 4))
                y0 = min(y0, max(0, hs[i] - WIN_H))
                self.refs.append(("slice", (files[i], y0, w), 0))
            n_ep_done += 1
            if NON_TTY and n_ep_done % 10 == 0:
                print(f"  [dataset] {n_ep_done}/{len(eps)} episodes done (refs={len(self.refs)})", flush=True)
        rng.shuffle(self.refs)

    def __len__(self):
        return len(self.refs)

    def _load(self, path, w):
        im = imread(path)
        if im is None:
            im = np.zeros((IN_SIZE + 50, w, 3), dtype=np.uint8)
        if im.shape[1] != w:
            im = cv2.resize(im, (w, im.shape[0]))
        return im

    def __getitem__(self, idx):
        typ, ref, label = self.refs[idx]
        if typ == "pos":
            a, b, w = self._load(ref[0], ref[2]), self._load(ref[1], ref[2]), ref[2]
            off = random.randint(-OFFSET_MAX, OFFSET_MAX)
            up_h = WIN_H // 2 + off
            down_h = WIN_H - up_h
            up = a[-up_h:] if a.shape[0] >= up_h else np.vstack([np.zeros((up_h - a.shape[0], w, 3), np.uint8), a])
            down = b[:down_h] if b.shape[0] >= down_h else np.vstack([b, np.zeros((down_h - b.shape[0], w, 3), np.uint8)])
            img = np.vstack([up, down])[:WIN_H]
        else:
            f, y0, w = ref
            a = self._load(f, w)
            y0 = min(y0, max(0, a.shape[0] - WIN_H))
            img = a[y0:y0 + WIN_H]
        img = cv2.resize(img, (IN_SIZE, IN_SIZE))
        return self.transform(img), torch.tensor(label, dtype=torch.long)


def train_cnn(train_eps, out_path, epochs, batch, lr, workers, seed=7, near_frac=0.5):
    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    console.print(Panel(
        f"[bold cyan]MobileNetV3-Small[/] | input [cyan]{IN_SIZE}x{IN_SIZE}[/] | "
        f"device=[green]{device.type.upper()} ({dev_name})[/] | "
        f"offset=[cyan]+-{OFFSET_MAX}px[/] near-neg=[cyan]{near_frac:.0%}[/]",
        title="[bold]TRAIN CNN BOUNDARY v2[/]", border_style="cyan"))

    with console.status("[bold green]Building dataset (loading images)...", spinner="dots"):
        ds = BoundaryDatasetV2(train_eps, seed=seed, near_frac=near_frac)
    n_pos = sum(1 for r in ds.refs if r[2] == 1)
    n_neg = len(ds.refs) - n_pos
    console.print(f"  [green]samples={len(ds)}[/]  pos=[cyan]{n_pos}[/]  neg=[magenta]{n_neg}[/]  "
                  f"episodes=[yellow]{len(train_eps)}[/]")

    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=True, persistent_workers=workers > 0)
    mdl = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    mdl.classifier[3] = nn.Linear(mdl.classifier[3].in_features, 2)
    mdl.to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = optim.AdamW(mdl.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    # ---- rich UI ----
    progress = Progress(
        TextColumn("[bold cyan]{task.description}[/]"),
        BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task_epoch = progress.add_task("[bold yellow]Epochs", total=epochs)
    task_batch = progress.add_task("[bold magenta]Batches", total=1)

    best_acc = 0.0
    best_ep = 0
    t_start = time.time()

    def gpu_str():
        if device.type != "cuda":
            return "CPU"
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        tot = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        try:
            util = torch.cuda.utilization()
            util_s = f"  util={util}%"
        except Exception:
            util_s = ""
        return f"{alloc:.2f}/{tot:.1f} GB" + util_s

    def render(epoch, loss, acc):
        t = Table(show_header=False, box=box.ROUNDED, expand=True, padding=(0, 1))
        t.add_column(style="bold yellow", justify="right")
        t.add_column(style="bold white")
        t.add_row("Epoch", f"{epoch} / {epochs}")
        t.add_row("Loss", f"{loss:.4f}" if loss is not None else "-")
        t.add_row("Acc (train)", f"{acc:.2f}%" if acc is not None else "-")
        t.add_row("Best acc", f"{best_acc:.2f}%  (epoch {best_ep})")
        t.add_row("GPU", gpu_str())
        t.add_row("Elapsed", f"{time.time()-t_start:.0f}s")
        return Group(Panel(progress, border_style="blue"),
                     Panel(t, title="[bold]Live metrics[/]", border_style="green"))

    with Live(render(0, None, None), console=console, refresh_per_second=8) as live:
        for epoch in range(epochs):
            mdl.train()
            loss_sum, cor, tot = 0.0, 0, 0
            progress.update(task_batch, total=len(dl), completed=0,
                            description=f"[bold magenta]Epoch {epoch+1}/{epochs}")
            for i, (x, y) in enumerate(dl):
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
                if NON_TTY:
                    if (i + 1) % 100 == 0:
                        print(f"  [E{epoch+1}] batch {i+1}/{len(dl)} "
                              f"loss={loss_sum/max(1,i+1):.3f} acc={cor/max(1,tot)*100:.1f}%",
                              flush=True)
                else:
                    progress.update(task_batch, advance=1)
                    live.update(render(epoch + 1, loss_sum / max(1, len(dl)),
                                       cor / max(1, tot) * 100))
            acc = cor / max(1, tot) * 100
            if acc > best_acc:
                best_acc, best_ep = acc, epoch + 1
            if NON_TTY:
                print(f"  [epoch {epoch+1} DONE] loss={loss_sum/max(1,len(dl)):.3f} "
                      f"acc={acc:.1f}%  best={best_acc:.1f}% (ep {best_ep})  saved={out_path}",
                      flush=True)
            else:
                progress.update(task_epoch, advance=1,
                                description=f"[bold yellow]Epoch {epoch+1}  "
                                            f"loss={loss_sum/max(1,len(dl)):.3f}  acc={acc:.1f}%")
                live.update(render(epoch + 1, loss_sum / max(1, len(dl)), acc))
            torch.save(mdl.state_dict(), out_path)
    console.print(Panel(
        f"[bold green]DONE[/]  best acc = [cyan]{best_acc:.2f}%[/] (epoch {best_ep})  "
        f"total [yellow]{time.time()-t_start:.0f}s[/]\n[bold]saved:[/] [green]{out_path}[/]",
        border_style="green"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--near-frac", type=float, default=0.5)
    ap.add_argument("--cnn-out", type=Path,
                    default=ROOT / "tools" / "habit_audit" / "poc" / "cnn_r4.pth")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    split = json.load(open(SPLIT, encoding="utf-8"))
    train_eps = split["train"]
    train_cnn(train_eps, args.cnn_out, args.epochs, args.batch, args.lr,
              args.workers, seed=args.seed, near_frac=args.near_frac)


if __name__ == "__main__":
    main()
