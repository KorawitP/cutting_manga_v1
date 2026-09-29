import sys, json, gc, time
from pathlib import Path
import numpy as np

sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import eval_both as E

# --- replicate the ORIGINAL app pipeline (EfficientNet-B0, ai_predictor.py) ---
import torch
import torch.nn as nn
import torchvision.models as models

SLICE_H = 256
TARGET_W = 720
STRIDE = 32

split = json.load(open(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json", encoding="utf-8"))
val = split["val"]

weights_path = r"D:\python\cutting_manga_v1\best_classifier.pth"
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)
model.load_state_dict(torch.load(weights_path, map_location=E.DEV, weights_only=True))
model.to(E.DEV).eval()
print(f"loaded {weights_path} -> EfficientNet-B0")

outdir = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\effnet_cache")
outdir.mkdir(exist_ok=True)

for e in val:
    name = f"{e['series']}__{e['episode']}"
    safe = name.replace("\\", "_").replace("/", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace('"', "_").replace("<", "_").replace(">", "_").replace("|", "_")
    np_path = outdir / f"{safe}.npz"
    if np_path.exists():
        continue
    ep = E.build_features_for_episode(Path(e["images"]))
    if ep is None:
        continue
    strip = ep["strip"]
    H = strip.shape[0]
    # 2. resize width -> 720 (height scaled proportionally)
    scale_ratio = TARGET_W / strip.shape[1]
    resized = E.cv2.resize(strip, (TARGET_W, int(H * scale_ratio)))
    RH = resized.shape[0]
    # 3. sliding window 256, stride 32
    y_starts = list(range(0, RH - SLICE_H, STRIDE))
    scores = []
    bs = 128
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(E.DEV)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(E.DEV)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(y_starts), bs):
            by = y_starts[i:i + bs]
            batch_np = np.empty((len(by), SLICE_H, 512, 3), dtype=np.float32)
            for k, sy in enumerate(by):
                win = resized[sy:sy + SLICE_H]
                batch_np[k] = E.cv2.resize(win, (512, SLICE_H), interpolation=E.cv2.INTER_LINEAR)
            batch_np *= 1.0 / 255.0
            bt = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous().to(E.DEV)
            bt = (bt - mean) / std
            with torch.amp.autocast("cuda" if E.DEV.type == "cuda" else "cpu"):
                out = model(bt)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            for k, sy in enumerate(by):
                scores.append((sy + SLICE_H // 2, float(probs[k])))
            del bt, out
    scores = np.array(scores, dtype=np.float32)  # (N,2): resized y center, safe prob
    yc_resized = scores[:, 0].astype(np.int32)
    sv = scores[:, 1]
    # 5. scale back to original coordinates
    ratio = H / RH
    yc_orig = (yc_resized * ratio).astype(np.int32)
    np.savez_compressed(np_path, yc=yc_orig, sv=sv, seams=ep["seams"], seg=ep["seg"],
                        density=ep["density"], H=H)
    gc.collect()
    print(f"{time.time()-t0:.0f}s {name} windows={len(scores)}")

print("effnet cache done")
