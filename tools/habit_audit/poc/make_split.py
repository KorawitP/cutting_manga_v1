import json, random, re
from pathlib import Path

SERIES_ROOT = Path(r"D:\python\YT-guide\project-data\series")
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "tools" / "habit_audit" / "poc" / "split.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SEED = 7
VAL_RATIO = 0.2


def natural_key(text):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"([0-9]+)", text)]


def list_episodes():
    episodes = []
    for series in sorted(SERIES_ROOT.iterdir(), key=lambda p: natural_key(p.name)):
        if not series.is_dir() or series.name.startswith("."):
            continue
        ep_dir = series / "episodes"
        if not ep_dir.is_dir():
            continue
        for ep in sorted(ep_dir.iterdir(), key=lambda p: natural_key(p.name)):
            img_dir = ep / "images"
            if img_dir.is_dir():
                episodes.append({"series": series.name, "episode": ep.name, "images": str(img_dir)})
    return episodes


def main():
    eps = list_episodes()
    random.Random(SEED).shuffle(eps)
    n_val = max(1, int(len(eps) * VAL_RATIO))
    val_eps, train_eps = eps[:n_val], eps[n_val:]
    payload = {
        "seed": SEED,
        "total": len(eps),
        "n_train": len(train_eps),
        "n_val": len(val_eps),
        "train": train_eps,
        "val": val_eps,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"total={len(eps)} train={len(train_eps)} val={len(val_eps)}")
    # per-series distribution in val
    from collections import Counter
    c = Counter(e["series"] for e in val_eps)
    print("val by series:", dict(c))
    print("saved:", OUT)


if __name__ == "__main__":
    main()
