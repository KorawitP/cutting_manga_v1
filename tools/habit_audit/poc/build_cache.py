import sys, json, gc, time
from pathlib import Path
import numpy as np

sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import eval_both as E

split = json.load(open(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json", encoding="utf-8"))
val = split["val"]
cnn = E.load_cnn(Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_r3.pth"))
outdir = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_cache_r3")
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
    strip_small = E.cv2.resize(strip, (512, H))
    strip_rgb = E.cv2.cvtColor(strip_small, E.cv2.COLOR_BGR2RGB)
    mean = E.torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(E.DEV)
    std = E.torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(E.DEV)
    y_centers, score_vals = [], []
    bs = 32
    CHUNK = 8192
    t0 = time.time()
    for chunk_start in range(0, H - E.WIN_H, CHUNK):
        chunk_end = min(H, chunk_start + CHUNK + E.WIN_H)
        arr = strip_rgb[chunk_start:chunk_end]
        t_chunk = E.torch.from_numpy(arr).permute(2, 0, 1).float().to(E.DEV) / 255.0
        t_chunk = (t_chunk - mean) / std
        batch_t, batch_y = [], []
        for sy in range(chunk_start, min(chunk_start + CHUNK, H - E.WIN_H, chunk_end - E.WIN_H), E.STRIDE):
            local = sy - chunk_start
            batch_t.append(t_chunk[:, local:local + E.WIN_H, :])
            batch_y.append(sy)
            if len(batch_t) == bs:
                bb = E.torch.stack(batch_t)
                with E.torch.no_grad():
                    probs = E.torch.softmax(cnn(bb), dim=1)[:, 1].cpu().numpy()
                for i, p in enumerate(probs):
                    y_centers.append(int(batch_y[i] + E.WIN_H // 2))
                    score_vals.append(float(p))
                batch_t, batch_y = [], []
        if batch_t:
            bb = E.torch.stack(batch_t)
            with E.torch.no_grad():
                probs = E.torch.softmax(cnn(bb), dim=1)[:, 1].cpu().numpy()
            for i, p in enumerate(probs):
                y_centers.append(int(batch_y[i] + E.WIN_H // 2))
                score_vals.append(float(p))
        del t_chunk
        E.torch.cuda.empty_cache()
    np.savez_compressed(np_path, yc=np.array(y_centers, dtype=np.int32),
                        sv=np.array(score_vals, dtype=np.float32),
                        seams=ep["seams"], seg=ep["seg"],
                        density=ep["density"], H=H)
    gc.collect()
    print(f"{time.time()-t0:.0f}s {name} windows={len(y_centers)}")

print("cache done")
