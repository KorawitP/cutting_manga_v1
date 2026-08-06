import sys, json, time, gc
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import eval_both as E

split = json.load(open(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json", encoding="utf-8"))
val = split["val"]
cnn = E.load_cnn(Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_r1.pth"))

# run CNN inference once, cache per-episode score arrays
cache = {}
for e in val:
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
    cache[e["episode"]] = {
        "yc": np.array(y_centers, dtype=np.int32),
        "sv": np.array(score_vals, dtype=np.float32),
        "seams": ep["seams"], "seg": ep["seg"], "density": ep["density"], "H": H,
    }
    gc.collect()

print(f"Cached {len(cache)} episodes")

def run_threshold(th, smooth_w=31):
    tot_hit, tot_real, tot_prec, tot_pred = 0, 0, 0, 0
    for name, c in cache.items():
        sv_sm = np.convolve(c["sv"], np.ones(smooth_w) / smooth_w, mode="same")
        dist = max(1, 30 // E.STRIDE // 2)
        pk, _ = find_peaks(sv_sm, height=th, distance=dist)
        peaks = c["yc"][pk]
        peaks = E.refine_peaks(peaks, c["density"], c["H"])
        r = E.eval_episode(peaks, c["seams"], c["seg"])
        if not r:
            continue
        rec, prec, np_, ns = r
        tot_hit += rec * ns; tot_real += ns
        tot_prec += prec * np_; tot_pred += np_
    return tot_hit / max(1, tot_real) * 100, tot_prec / max(1, tot_pred) * 100

print("threshold sweep:")
for th in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
    r, p = run_threshold(th)
    print(f"  th={th:.2f} recall={r:.1f}% precision={p:.1f}%")

print("smooth window sweep at th=0.3:")
for w in [11, 21, 31, 41, 61]:
    r, p = run_threshold(0.3, w)
    print(f"  w={w} recall={r:.1f}% precision={p:.1f}%")
