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

def merge_strip(ep_dir, max_slices, max_orig_h=0):
    files = sorted([p for p in ep_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                   key=lambda p: natural_key(p.name))
    imgs, hs = [], []
    cum = 0
    for p in files:
        im = imread_color(p)
        if im is None: continue
        imgs.append(im); hs.append(im.shape[0])
        cum += im.shape[0]
        if max_orig_h and cum >= max_orig_h: break
        if max_slices and len(imgs) >= max_slices: break
    if not imgs: return None
    w = min(im.shape[1] for im in imgs)
    for i, im in enumerate(imgs):
        if im.shape[1] != w: imgs[i] = cv2.resize(im, (w, im.shape[0]))
    strip = np.vstack(imgs)
    H = strip.shape[0]
    seams = np.cumsum(hs, dtype=np.int32)[:-1]
    # build features for GBM
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
        else: i += 1
    density = np.convolve(edge, np.ones(41)/41, mode="same")
    dist = np.arange(H)/max(1,H-1)
    feats = np.stack([std,white,edge,is_g.astype(np.float32),gw,density,dist],axis=1)
    fn = feats.copy()
    for c in [0,1,2,5]:
        mu,sd = fn[:,c].mean(),fn[:,c].std()+1e-6
        fn[:,c] = (fn[:,c]-mu)/sd
    seg = np.array([bool(is_g[max(0,s-2):s+3].mean()>=0.6) for s in seams])
    return {"strip":strip, "feats":fn, "seams":seams, "seg":seg, "density":density, "H":H}

def refine_peaks(peaks, density, H):
    out=[]
    for pk in peaks:
        lo=max(0,int(pk)-20); hi=min(H-1,int(pk)+20)
        out.append(lo+int(np.argmin(density[lo:hi+1])))
    return np.array(out,dtype=np.int32)

# ---- Models ----
def load_cnn(path):
    mdl = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    mdl.classifier[3] = nn.Linear(mdl.classifier[3].in_features, 2)
    mdl.load_state_dict(torch.load(path, map_location=DEV))
    return mdl.eval().to(DEV)

def run_gbm(ep, gbm, th=0.15, sw=31):
    probs = gbm.predict_proba(ep["feats"])[:,1]
    score = probs.copy(); score[ep["feats"][:,4]<5] *= 0.3
    score = np.convolve(score, np.ones(sw)/sw, mode="same")
    pk, _ = find_peaks(score, height=th, distance=max(1,30//2))
    return refine_peaks(pk, ep["density"], ep["H"])

def run_cnn(ep, model, th=0.35, sw=11):
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
                with torch.no_grad(): p=torch.softmax(model(bb),dim=1)[:,1].cpu().numpy()
                for i,pr in enumerate(p): yc.append(int(by[i]+WIN_H//2)); sv.append(float(pr))
                bt,by=[],[]
        if bt:
            bb=torch.stack(bt)
            with torch.no_grad(): p=torch.softmax(model(bb),dim=1)[:,1].cpu().numpy()
            for i,pr in enumerate(p): yc.append(int(by[i]+WIN_H//2)); sv.append(float(pr))
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

# ===== MAIN =====
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--series", default="i-will-raise-this-family-to-greatness")
ap.add_argument("--episode", default="ตอนที่ 44")
ap.add_argument("--max-h", type=int, default=25000)
ap.add_argument("--out", default="compare_app.jpg")
args = ap.parse_args()

ep_dir = Path(r"D:\python\YT-guide\project-data\series") / args.series / "episodes" / args.episode / "images"
if not ep_dir.is_dir():
    raise FileNotFoundError(f"{ep_dir}")

print(f"Merging first {args.max_h} original px of {args.series} / {args.episode}...")
ep = merge_strip(ep_dir, max_slices=99, max_orig_h=args.max_h)
print(f"Strip: {ep['strip'].shape[1]}x{ep['strip'].shape[0]} seams={len(ep['seams'])}")

gbm = joblib.load(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\gbm_r1.joblib")
cnn = load_cnn(Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc\cnn_r3.pth"))

t0=time.time(); pred_g = run_gbm(ep, gbm, 0.15, 31); tg=time.time()-t0
t0=time.time(); pred_c = run_cnn(ep, cnn, 0.35, 11); tc=time.time()-t0
rg = eval_pred(pred_g, ep["seams"], ep["seg"])
rc = eval_pred(pred_c, ep["seams"], ep["seg"])
print(f"GBM recall={rg:.0f}% ({len(pred_g)} cuts, {tg:.1f}s) | CNN recall={rc:.0f}% ({len(pred_c)} cuts, {tc:.1f}s)")

# Render: scale to 400px wide each
OUT_W = 400  
H = ep["strip"].shape[0]
scale = OUT_W / ep["strip"].shape[1]
OUT_H = int(H * scale)

def render_side(strip, seams, pred, label, color):
    img = cv2.resize(strip, (OUT_W, OUT_H))
    for s in seams: cv2.line(img, (0,int(s*scale)), (OUT_W,int(s*scale)), (0,255,0), 1)
    for p in pred: cv2.line(img, (0,int(p*scale)), (OUT_W,int(p*scale)), color, 2)
    cv2.putText(img, label, (4, OUT_H-14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return img

img_g = render_side(ep["strip"], ep["seams"], pred_g,
    f"GBM  recall={rg:.0f}%  cuts={len(pred_g)}  {tg:.1f}s", (0,0,255))
img_c = render_side(ep["strip"], ep["seams"], pred_c,
    f"CNN  recall={rc:.0f}%  cuts={len(pred_c)}  {tc:.1f}s", (255,100,100))

# top bar
bar = np.zeros((30, OUT_W*2+4, 3), dtype=np.uint8)
cv2.putText(bar, f"{args.series} {args.episode}  (first {args.max_h}px)  |  Green=seams  Red=predicted",
            (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

canvas = np.hstack([img_g, np.zeros((OUT_H,4,3),dtype=np.uint8), img_c])
canvas = np.vstack([bar, canvas])

out = Path(r"D:\python\cutting_manga_v1\tools\habit_audit\poc") / args.out
cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tofile(out)
print(f"saved: {out}  ({canvas.shape[1]}x{canvas.shape[0]})")
