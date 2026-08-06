import json, random, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import train_r4 as T

# ---- experiment: larger context 768px (vs 512) ----
T.WIN_H = 768
T.IN_SIZE = 768
T.OFFSET_MAX = 64
T.NEAR_MIN = 80
T.NEAR_MAX = 420

ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT / "tools" / "habit_audit" / "poc" / "split.json"
OUT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_768.pth")


def main():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    split = json.load(open(SPLIT, encoding="utf-8"))
    print(f"WIN_H={T.WIN_H} IN_SIZE={T.IN_SIZE} offset<=+-{T.OFFSET_MAX} "
          f"near={T.NEAR_MIN}-{T.NEAR_MAX} | train eps={len(split['train'])}")
    T.train_cnn(split["train"], OUT, epochs=8, batch=16, lr=1e-4, workers=4,
                seed=7, near_frac=0.5)


if __name__ == "__main__":
    main()
