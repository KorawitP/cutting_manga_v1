from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RESULT_DIR_NAME = "_content_filter_result"


@dataclass
class FilterConfig:
    weights_path: Path
    discard_threshold: float = 0.95
    keep_threshold: float = 0.80
    image_size: int = 512
    action: str = "copy"
    existing: str = "skip"


@dataclass
class FilterSummary:
    episode_dir: Path
    output_dir: Path
    keep: int = 0
    discard_suggested: int = 0
    review: int = 0
    error: int = 0
    skipped: bool = False


class LetterboxSquare:
    def __init__(self, size: int, fill=(255, 255, 255)):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")

        img.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
        pad_w = self.size - img.width
        pad_h = self.size - img.height
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        return ImageOps.expand(img, border=(left, top, right, bottom), fill=self.fill)


def natural_sort_key(path: Path):
    import re

    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", path.name)]


def image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=natural_sort_key,
    )


def discover_episode_dirs(root: Path) -> list[Path]:
    """Return folders that directly contain images, ignoring previous result folders."""
    if image_files(root):
        return [root]

    direct = [
        p for p in root.iterdir()
        if p.is_dir() and p.name != RESULT_DIR_NAME and image_files(p)
    ]
    if direct:
        return sorted(direct, key=natural_sort_key)

    leaf_dirs = []
    for folder in root.rglob("*"):
        if not folder.is_dir() or RESULT_DIR_NAME in folder.parts:
            continue
        if image_files(folder):
            leaf_dirs.append(folder)
    return sorted(leaf_dirs, key=lambda p: str(p).lower())


def unique_destination(dst: Path) -> Path:
    if not dst.exists():
        return dst

    parent = dst.parent
    stem = dst.stem
    suffix = dst.suffix
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx:03d}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def route_prediction(discard_prob: float, keep_prob: float, config: FilterConfig) -> str:
    if discard_prob >= config.discard_threshold:
        return "discard_suggested"
    if keep_prob >= config.keep_threshold:
        return "keep"
    return "review"


class ContentFilterPredictor:
    def __init__(self, config: FilterConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            LetterboxSquare(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.model = self._load_model(config.weights_path)

    def _load_model(self, weights_path: Path):
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        model.to(self.device)
        model.eval()
        return model

    def classify_image(self, path: Path) -> tuple[str, float, float]:
        with Image.open(path) as img:
            tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].detach().cpu()

        discard_prob = float(probs[0])
        keep_prob = float(probs[1])
        pred_class = "0_discard" if discard_prob >= keep_prob else "1_keep"
        return pred_class, discard_prob, keep_prob

    def process_episode(self, episode_dir: Path) -> FilterSummary:
        output_dir = episode_dir / RESULT_DIR_NAME
        if output_dir.exists():
            if self.config.existing == "skip":
                return FilterSummary(episode_dir=episode_dir, output_dir=output_dir, skipped=True)
            if self.config.existing == "overwrite":
                shutil.rmtree(output_dir)

        keep_dir = output_dir / "keep"
        discard_dir = output_dir / "discard_suggested"
        review_dir = output_dir / "review"
        for folder in (keep_dir, discard_dir, review_dir):
            folder.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "predictions.csv"
        files = image_files(episode_dir)
        summary = FilterSummary(episode_dir=episode_dir, output_dir=output_dir)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "route", "pred_class", "discard_prob", "keep_prob", "error"])

            for path in tqdm(files, desc=episode_dir.name, leave=False):
                try:
                    pred_class, discard_prob, keep_prob = self.classify_image(path)
                    route = route_prediction(discard_prob, keep_prob, self.config)
                    dst_root = {
                        "keep": keep_dir,
                        "discard_suggested": discard_dir,
                        "review": review_dir,
                    }[route]
                    dst = unique_destination(dst_root / path.name)

                    if self.config.action == "move":
                        shutil.move(str(path), str(dst))
                    else:
                        shutil.copy2(path, dst)

                    setattr(summary, route, getattr(summary, route) + 1)
                    writer.writerow([path.name, route, pred_class, f"{discard_prob:.6f}", f"{keep_prob:.6f}", ""])
                except Exception as exc:
                    summary.error += 1
                    writer.writerow([path.name, "error", "", "", "", str(exc)])

        return summary

    def process_episodes(self, episode_dirs: Iterable[Path]) -> list[FilterSummary]:
        return [self.process_episode(episode_dir) for episode_dir in episode_dirs]
