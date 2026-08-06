import json, time, re, cv2, torch, numpy as np, joblib
from pathlib import Path
import torchvision.models as models
import torch.nn as nn
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[0]
TOL_N, TOL_H = 100, 200
DEV = torch.device("cuda")
STRIDE = 16
WIN_H = 512
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def natural_key(text):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"([0-9]+)", text)]

def imread_color(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def build_features(ep_dir):
    files = sorted([p for p in ep_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                   key=lambda p: natural_key(p.name))
    imgs, hs = [], []
    for p in files:
        im = imread_color(p)
        if im is None: continue
        hs.append(im.shape[0]); imgs.append(im)
    if not imgs: return None
    w = min(im.shape[1] for im in imgs)
    for i, im in enumerate(imgs):
        if im.shape[1] != w: imgs[i] = cv2.resize(im, (w, im.shape[0]))
    strip = np.vstack(imgs)
    H = strip.shape[0]
    seams = np.cumsum(hs, dtype=np.int32)[:-1]

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    sobel = cv2.addWeighted(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), 1,
                            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3), 1, 0)
    sobel = cv2.convertScaleAbs(sobel).astype(np.float32)
    std = gray.std(axis=1).astype(np.float32)
    white = (gray >= 230).mean(axis=1).astype(np.float32)
    edge = sobel.mean(axis=1).astype(np.float32)
    is_g = ((std < 12) | (white >= 0.97)) & (edge < 4)
    gw = np.zeros(H, dtype=np.int32); i = 0
    while i < H:
        if is_g[i]:
            j = i
            while j < H and is_g[j]: j += 1
            gw[i:j] = j - i; i = j
        else:
            i += 1
    density = np.convolve(edge, np.ones(41)/41, mode="same")
    dist_top = np.arange(H)/max(1,H-1)
    feats = np.stack([std,white,edge,is_g.astype(np.float32),gw.astype(np.float32),density,dist_top],axis=1)
    feats_norm = feats.astype(np.float32).copy()
    for c in [0,1,2,5]:
        mu,sd = feats_norm[:,c].mean(),feats_norm[:,c].std()+1e-6
        feats_norm[:,c] = (feats_norm[:,c]-mu)/sd
    seg = np.array([bool(is_g[max(0,s-2):s+3].mean()>=0.6) for s in seams])
    return {"strip":strip, "feats":feats_norm, "seams":seams, "seg":seg, "density":density, "H":H}

def refine_peaks(peaks, density, H):
    out=[]
    for pk in peaks:
        lo=max(0,int(pk)-20); hi=min(H-1,int(pk)+20)
        out.append(lo+int(np.argmin(density[lo:hi+1])))
    return np.array(out,dtype=np.int32)

def run_gbm(ep, gbm, th=0.10, sw=31):
    probs = gbm.predict_proba(ep["feats"])[:,1]
    score = probs.copy(); score[ep["feats"][:,4]<5] *= 0.3
    score = np.convolve(score, np.ones(sw)/sw, mode="same")
    pk, _ = find_peaks(score, height=th, distance=max(1,30//2))
    return refine_peaks(pk, ep["density"], ep["H"])

def load_cnn(path):
    mdl = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    mdl.classifier[3] = nn.Linear(mdl.classifier[3].in_features, 2)
    mdl.load_state_dict(torch.load(path, map_location=DEV))
    return mdl.eval().to(DEV)

def run_cnn(ep, model, th=0.30, sw=11):
    H = ep["H"]
    strip_small = cv2.resize(ep["strip"], (512, H))
    strip_rgb = cv2.cvtColor(strip_small, cv2.COLOR_BGR2RGB)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEV)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEV)
    yc, sv = [], []
    bs, CHUNK = 32, 8192
    for cs in range(0, H-WIN_H, CHUNK):
        ce = min(H, cs+CHUNK+WIN_H)
        arr = strip_rgb[cs:ce]
        t = torch.from_numpy(arr).permute(2,0,1).float().to(DEV)/255
        t = (t-mean)/std
        bt, by = [], []
        for sy in range(cs, min(cs+CHUNK, H-WIN_H, ce-WIN_H), STRIDE):
            bt.append(t[:, sy-cs:sy-cs+WIN_H, :]); by.append(sy)
            if len(bt)==bs:
                bb=torch.stack(bt)
                with torch.no_grad(): probs=torch.softmax(model(bb),dim=1)[:,1].cpu().numpy()
                for i,p in enumerate(probs): yc.append(int(by[i]+WIN_H//2)); sv.append(float(p))
                bt,by=[],[]
        if bt:
            bb=torch.stack(bt)
            with torch.no_grad(): probs=torch.softmax(model(bb),dim=1)[:,1].cpu().numpy()
            for i,p in enumerate(probs): yc.append(int(by[i]+WIN_H//2)); sv.append(float(p))
        del t; torch.cuda.empty_cache()
    yc = np.array(yc,dtype=np.int32); sv = np.array(sv,dtype=np.float32)
    sv_sm = np.convolve(sv, np.ones(sw)/sw, mode="same")
    pk, _ = find_peaks(sv_sm, height=th, distance=max(1, 30//STRIDE//2))
    return refine_peaks(yc[pk], ep["density"], H)

def eval_pred(pred, seams, seg):
    hit=0; used=set()
    for i,s in enumerate(seams):
        tol=TOL_N if seg[i] else TOL_H
        d=np.abs(pred-s)
        if len(d)==0: continue
        best=int(np.argmin(d))
        if d[best]<=tol and best not in used: hit+=1; used.add(best)
    return hit/len(seams)*100

def match_pred_to_seam(pred, seams, seg):
    """return distance of nearest pred to each seam, or -1 if none within tol"""
    results = []
    used = set()
    for i, s in enumerate(seams):
        tol = TOL_N if seg[i] else TOL_H
        d = np.abs(pred - s)
        if len(d) == 0:
            results.append(-1)
            continue
        best = int(np.argmin(d))
        if d[best] <= tol and best not in used:
            results.append(d[best])
            used.add(best)
        else:
            results.append(-1)
    return results

# ---- main ----
ep_dir = Path(r"D:\python\YT-guide\project-data\series\i-will-raise-this-family-to-greatness\episodes\ตอนที่ 44\images")
gbm_path = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\gbm_r1.joblib")
cnn_path = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_r3.pth")

print("Loading...")
ep = build_features(ep_dir)
print(f"H={ep['H']} seams={len(ep['seams'])} hard={int((~ep['seg']).sum())}")
gbm = joblib.load(gbm_path)
cnn = load_cnn(cnn_path)

t0=time.time(); pred_gbm = run_gbm(ep, gbm, 0.10, 31); tg=time.time()-t0
t0=time.time(); pred_cnn = run_cnn(ep, cnn, 0.30, 11); tc=time.time()-t0
rg = eval_pred(pred_gbm, ep["seams"], ep["seg"])
rc = eval_pred(pred_cnn, ep["seams"], ep["seg"])
print(f"GBM recall={rg:.1f}% ({tg:.1f}s) | CNN recall={rc:.1f}% ({tc:.1f}s)")

# compute per-seam distances for both models
dg = match_pred_to_seam(pred_gbm, ep["seams"], ep["seg"])
dc = match_pred_to_seam(pred_cnn, ep["seams"], ep["seg"])

# pick interesting zoom positions: evenly spaced + where models differ most
diff = np.abs(np.array(dg) - np.array(dc))
# replace -1 with large value for sorting
diffs = np.where((np.array(dg) >= 0) | (np.array(dc) >= 0), diff, 0)
# pick 8: 4 best-aligning + 4 most-different
n_show = 8
indices = np.linspace(0, len(ep["seams"])-1, n_show).astype(int)

ZOOM = 400  # px around each seam
OUT_W = 500
strip = ep["strip"]
H = ep["H"]

rows = []
for si in indices:
    sy = int(ep["seams"][si])
    top = max(0, sy - ZOOM)
    bot = min(H, sy + ZOOM)
    crop = strip[top:bot]
    # render for GBM and CNN
    def render_crop(crop, seams_abs, preds_abs, focus_y, label, color=(0,0,255)):
        hh = crop.shape[0]
        img = cv2.resize(crop, (OUT_W, int(hh * OUT_W / crop.shape[1])))
        scale_y = img.shape[0] / hh
        # draw all real seams in this region
        for s in seams_abs:
            if top <= s <= bot:
                y = int((s - top) * scale_y)
                cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 1)
        # draw all predictions in this region
        for p in preds_abs:
            if top <= p <= bot:
                y = int((p - top) * scale_y)
                cv2.line(img, (0, y), (img.shape[1], y), color, 2)
        # label
        cv2.putText(img, label, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        # focus seam highlight
        fy = int((focus_y - top) * scale_y)
        cv2.rectangle(img, (0, fy-4), (img.shape[1], fy+4), (255, 255, 0), 1)
        return img

    img_g = render_crop(crop, ep["seams"], pred_gbm, sy, f"GBM seam={si} hit={dg[si]}px")
    img_c = render_crop(crop, ep["seams"], pred_cnn, sy, f"CNN seam={si} hit={dc[si]}px", (255,100,100))

    hh = max(img_g.shape[0], img_c.shape[0])
    img_g = cv2.copyMakeBorder(img_g, 0, hh-img_g.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20,20,20))
    img_c = cv2.copyMakeBorder(img_c, 0, hh-img_c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20,20,20))
    rows.append(np.hstack([img_g, img_c]))

canvas = np.vstack(rows)

# header bar
header = np.zeros((40, canvas.shape[1], 3), dtype=np.uint8)
cv2.putText(header, f"GBM recall={rg:.1f}% {tg:.1f}s   |   CNN recall={rc:.1f}% {tc:.1f}s   |   Green=real seams   Red=predicted   Yellow=focus",
            (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
canvas = np.vstack([header, canvas])

out = r"D:\python\cutting_manga_v1\tools\habit_audit\poc\compare_viz2.jpg"
cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(out)
print(f"saved: {out}  size: {canvas.shape[1]}x{canvas.shape[0]}")
