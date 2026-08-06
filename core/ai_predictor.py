import os
import cv2
import numpy as np
try:
    import importlib.util
    torch_spec = importlib.util.find_spec("torch")
    TORCH_AVAILABLE = torch_spec is not None
except ImportError:
    TORCH_AVAILABLE = False

from PIL import Image
from scipy.signal import find_peaks


class AIPredictor:
    """AI ตัดภาพ v5 (MobileNetV3-Small, window 512, DP count-free)

    interface เดิม: load_model() / predict_cut_points(pil_image, ...) -> (ok, cut_ys, msg)
    cut_ys = พิกัด y (scale เดิมของภาพที่ส่งเข้า)
    """

    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.device = None
        self.model = None
        self.is_loaded = False
        self.arch = None  # "v5" (MobileNetV3) หรือ "effnet" (EfficientNetB0 เดิม)

        # v5 config
        self.window_h = 512
        self.input_width = 512
        self.stride = 16
        self.smooth = 15
        self.a_den = 0.75
        self.min_dist = 400
        self.lam = 0.8
        self.refine_r = 10

        # fallback (EfficientNet เดิม)
        self.slice_height = 256
        self.target_width = 720
        self.min_cut_distance = 300
        self.safe_threshold = 0.90

        self.mean = None
        self.std = None

    def _init_torch(self):
        if not TORCH_AVAILABLE:
            return False
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)
            return True
        except Exception as e:
            print(f"Error initializing PyTorch environment: {e}")
            return False

    def load_model(self):
        if not self._init_torch():
            return False, "ไม่สามารถโหลดไลบรารี PyTorch ได้ (DLL Error หรือไม่ได้ติดตั้ง)"
        if not os.path.exists(self.weights_path):
            return False, f"ไม่พบไฟล์โมเดลที่: {self.weights_path}"
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
            state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
            # ลอง MobileNetV3 (v5) ก่อน
            mdl = models.mobilenet_v3_small(weights=None)
            mdl.classifier[3] = nn.Linear(mdl.classifier[3].in_features, 2)
            try:
                mdl.load_state_dict(state)
                self.model = mdl
                self.arch = "v5"
            except Exception:
                # fallback EfficientNetB0 (โมเดลเก่า)
                mdl2 = models.efficientnet_b0(weights=None)
                in_f = mdl2.classifier[1].in_features
                mdl2.classifier[1] = nn.Linear(in_f, 2)
                mdl2.load_state_dict(state)
                self.model = mdl2
                self.arch = "effnet"
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            return True, "โหลดโมเดลสำเร็จ"
        except Exception as e:
            return False, f"โหลดโมเดลล้มเหลว: {str(e)}"

    # ---------- v5 pipeline ----------

    def _make_batch_tensor_v5(self, img, y_batch):
        import torch
        batch_np = np.empty((len(y_batch), self.window_h, self.input_width, 3), dtype=np.float32)
        for idx, start_y in enumerate(y_batch):
            window = img[start_y:start_y + self.window_h, :]
            resized = cv2.resize(window, (self.input_width, self.window_h), interpolation=cv2.INTER_LINEAR)
            batch_np[idx] = resized
        batch_np *= 1.0 / 255.0
        batch_stack = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous().to(self.device)
        return (batch_stack - self.mean) / self.std

    @staticmethod
    def _dp_free(score, min_dist, lam, gap):
        M = len(score)
        idxgap = max(1, int(np.ceil(min_dist / gap)))
        INF = -1e18
        dp = np.full(M, INF)
        bp = np.full(M, -1, dtype=np.int64)
        bv, bi = INF, -1
        for i in range(M):
            j = i - idxgap
            if j >= 0 and dp[j] > bv:
                bv, bi = dp[j], j
            prev = bv if bv > 0 else 0.0
            dp[i] = score[i] - lam + prev
            bp[i] = bi if bv > 0 else -1
        last = int(np.argmax(dp))
        idxs = [last]
        while bp[last] != -1:
            last = int(bp[last])
            idxs.append(last)
        return np.sort(np.array(idxs, dtype=np.int64))

    def _predict_v5(self, pil_image, progress_callback, cancel_callback):
        import torch
        open_cv_image = np.array(pil_image)
        if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:
            img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        elif len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 4:
            img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGR)
        else:
            img = open_cv_image.copy()

        h, w = img.shape[:2]
        if w != self.input_width:
            img = cv2.resize(img, (self.input_width, int(h * self.input_width / w)))
        total_h = img.shape[0]

        y_starts = list(range(0, total_h - self.window_h, self.stride))
        if not y_starts:
            return True, [], "ภาพสั้นเกินไป ไม่จำเป็นต้องหาจุดตัด"

        # density (บน working strip) สำหรับ density prior
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel = cv2.addWeighted(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), 1,
                                cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3), 1, 0)
        sobel = cv2.convertScaleAbs(sobel).astype(np.float32)
        density = np.convolve(sobel.mean(axis=1), np.ones(41, dtype=np.float32) / 41, mode="same")

        y_centers, scores = [], []
        batch_size = 128 if self.device.type == "cuda" else 64
        total_steps = len(y_starts)
        step_count = 0
        with torch.inference_mode():
            bs = 0
            while bs < total_steps:
                if cancel_callback and cancel_callback():
                    return False, [], "ยกเลิกการวิเคราะห์ AI"
                batch_y = y_starts[bs:bs + batch_size]
                try:
                    batch_stack = self._make_batch_tensor_v5(img, batch_y)
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(batch_stack)
                    probs = torch.softmax(outputs, dim=1)
                    p1 = probs[:, 1].cpu().numpy()
                    for i, p in enumerate(p1):
                        y_centers.append(int(batch_y[i] + self.window_h // 2))
                        scores.append(float(p))
                    step_count += len(batch_y)
                    bs += len(batch_y)
                    if progress_callback:
                        progress_callback(int((step_count / total_steps) * 80))
                    del batch_stack, outputs, probs
                except RuntimeError as exc:
                    if self.device.type == "cuda" and "out of memory" in str(exc).lower() and batch_size > 8:
                        torch.cuda.empty_cache()
                        batch_size = max(8, batch_size // 2)
                        continue
                    raise

        if not scores:
            return False, [], "ไม่สามารถสร้างผลการวิเคราะห์จากโมเดลได้"
        if progress_callback:
            progress_callback(90)

        yc = np.array(y_centers, dtype=np.int32)
        sv = np.array(scores, dtype=np.float32)
        svs = np.convolve(sv, np.ones(self.smooth) / self.smooth, mode="same")
        score = svs - self.a_den * (density[yc] / max(1.0, float(density.max())))
        idxs = self._dp_free(score, self.min_dist, self.lam, self.stride)
        cuts = yc[idxs].astype(np.int64)
        out = []
        for c in cuts:
            lo = max(0, int(c) - self.refine_r); hi = min(total_h - 1, int(c) + self.refine_r)
            out.append(lo + int(np.argmin(density[lo:hi + 1])))
        cut_ys = np.array(out, dtype=np.int32)

        # scale กลับสู่ภาพเดิม
        original_scale_ratio = pil_image.height / total_h
        original_cut_ys = [int(y * original_scale_ratio) for y in cut_ys]

        if progress_callback:
            progress_callback(100)
        return True, original_cut_ys, f"พบจุดตัด {len(original_cut_ys)} จุด โดย AI (v5)"

    # ---------- EfficientNet เดิม (backward compat) ----------

    def _make_batch_tensor(self, img, y_batch):
        import torch
        batch_np = np.empty((len(y_batch), self.slice_height, self.input_width, 3), dtype=np.float32)
        for idx, start_y in enumerate(y_batch):
            window = img[start_y:start_y + self.slice_height, :]
            resized = cv2.resize(window, (self.input_width, self.slice_height), interpolation=cv2.INTER_LINEAR)
            batch_np[idx] = resized
        batch_np *= 1.0 / 255.0
        batch_stack = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous().to(self.device)
        return (batch_stack - self.mean) / self.std

    def _predict_effnet(self, pil_image, progress_callback, cancel_callback):
        import torch
        open_cv_image = np.array(pil_image)
        if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:
            img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        elif len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 4:
            img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGR)
        else:
            img = open_cv_image.copy()
        h, w = img.shape[:2]
        scale_ratio = self.target_width / w
        if w != self.target_width:
            img = cv2.resize(img, (self.target_width, int(h * scale_ratio)))
        total_h = img.shape[0]
        y_starts = list(range(0, total_h - self.slice_height, self.stride))
        if not y_starts:
            return True, [], "ภาพสั้นเกินไป ไม่จำเป็นต้องหาจุดตัด"
        scores = []
        batch_size = 128 if self.device and self.device.type == "cuda" else 64
        total_steps = len(y_starts)
        step_count = 0
        with torch.inference_mode():
            batch_start = 0
            while batch_start < total_steps:
                if cancel_callback and cancel_callback():
                    return False, [], "ยกเลิกการวิเคราะห์ AI"
                batch_y = y_starts[batch_start:batch_start + batch_size]
                try:
                    batch_stack = self._make_batch_tensor(img, batch_y)
                    if self.device.type == "cuda" and hasattr(torch.amp, "autocast"):
                        with torch.amp.autocast("cuda"):
                            outputs = self.model(batch_stack)
                    else:
                        outputs = self.model(batch_stack)
                    probs = torch.softmax(outputs, dim=1)
                    safe_probs = probs[:, 1].cpu().numpy()
                    for i, p in enumerate(safe_probs):
                        scores.append((batch_y[i] + (self.slice_height // 2), p))
                    step_count += len(batch_y)
                    batch_start += len(batch_y)
                    if progress_callback:
                        progress_callback(int((step_count / total_steps) * 80))
                    del batch_stack, outputs, probs
                except RuntimeError as exc:
                    if self.device.type == "cuda" and "out of memory" in str(exc).lower() and batch_size > 8:
                        torch.cuda.empty_cache()
                        batch_size = max(8, batch_size // 2)
                        continue
                    raise
        if not scores:
            return False, [], "ไม่สามารถสร้างผลการวิเคราะห์จากโมเดลได้"
        if progress_callback:
            progress_callback(90)
        y_coords = np.array([s[0] for s in scores])
        safe_scores = np.array([s[1] for s in scores])
        distance_in_array = self.min_cut_distance // self.stride
        peaks, _ = find_peaks(safe_scores, height=self.safe_threshold, distance=distance_in_array)
        cut_ys = y_coords[peaks]
        original_scale_ratio = pil_image.height / total_h
        original_cut_ys = [int(y * original_scale_ratio) for y in cut_ys]
        if progress_callback:
            progress_callback(100)
        return True, original_cut_ys, f"พบจุดตัด {len(original_cut_ys)} จุด โดย AI"

    def predict_cut_points(self, pil_image, progress_callback=None, cancel_callback=None):
        if not self.is_loaded:
            success, msg = self.load_model()
            if not success:
                return False, [], msg
        try:
            if self.arch == "v5":
                return self._predict_v5(pil_image, progress_callback, cancel_callback)
            return self._predict_effnet(pil_image, progress_callback, cancel_callback)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, [], f"การวิเคราะห์ล้มเหลว: {str(e)}"
