import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, r"D:\python\cutting_manga_v1\tools\habit_audit\poc")
import eval_both as E
import eval_dp2 as D

MODEL = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_r4.pth")
SPLIT = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\split.json")
SW, A_DEN, MD, RR = 15, 1.0, 1500, 10

def safe(name):
    return (name.replace("\\", "_").replace("/", "_").replace(":", "_")
            .replace("*", "_").replace("?", "_").replace('"', "_")
            .replace("<", "_").replace(">", "_").replace("|", "_"))

def run_cnn_episode(ep, model):
    """Full CNN scoring (stride 16, 512px) returns yc, sv_sm"""
    H = ep["strip"].shape[0]
    strip = ep["strip"]
    strip_small = E.cv2.resize(strip, (512, H))
    strip_rgb = E.cv2.cvtColor(strip_small, E.cv2.COLOR_BGR2RGB)
    mean = E.torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(E.DEV)
    std = E.torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(E.DEV)
    y_centers, scores = [], []
    bs, CHUNK = 32, 8192
    for cs in range(0, H - E.WIN_H, CHUNK):
        ce = min(H, cs + CHUNK + E.WIN_H)
        arr = strip_rgb[cs:ce]
        t = E.torch.from_numpy(arr).permute(2, 0, 1).float().to(E.DEV) / 255.0
        t = (t - mean) / std
        bt, by = [], []
        for sy in range(cs, min(cs + CHUNK, H - E.WIN_H, ce - E.WIN_H), E.STRIDE):
            bt.append(t[:, sy - cs:sy - cs + E.WIN_H, :]); by.append(sy)
            if len(bt) == bs:
                bb = E.torch.stack(bt)
                with E.torch.inference_mode():
                    p = E.torch.softmax(model(bb), dim=1)[:, 1].cpu().numpy()
                y_centers.extend(int(y + E.WIN_H // 2) for y in by)
                scores.extend(map(float, p))
                bt, by = [], []
        if bt:
            bb = E.torch.stack(bt)
            with E.torch.inference_mode():
                p = E.torch.softmax(model(bb), dim=1)[:, 1].cpu().numpy()
            y_centers.extend(int(y + E.WIN_H // 2) for y in by)
            scores.extend(map(float, p))
        del t
        E.torch.cuda.empty_cache()
    yc = np.array(y_centers, dtype=np.int32)
    sv = np.array(scores, dtype=np.float32)
    sv_sm = np.convolve(sv, np.ones(SW) / SW, mode="same")
    return yc, sv_sm

def main():
    split = json.load(open(SPLIT, encoding="utf-8"))
    model = E.load_cnn(MODEL)
    # pick 3 val episodes
    picks = split["val"][:3]
    for e in picks:
        ep = E.build_features_for_episode(Path(e["images"]))
        if ep is None:
            continue
        t0 = time.time()
        yc, sv_sm = run_cnn_episode(ep, model)
        t_cnn = time.time() - t0
        t0 = time.time()
        score = sv_sm - A_DEN * (ep["density"][yc] / max(1.0, float(ep["density"].max())))
        cuts = D.dp_select(yc, score, len(ep["seams"]), MD)
        t_dp = time.time() - t0
        print(f"{e['series']} / {e['episode']:>10s} | H={ep['strip'].shape[0]:>7d}px "
              f"windows={len(yc):>6d} | CNN {t_cnn:5.1f}s + DP {t_dp*1000:5.0f}ms = "
              f"{t_cnn + t_dp:5.1f}s/total")

if __name__ == "__main__":
    main()
