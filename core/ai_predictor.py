import os
import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks

# พยายามตรวจสอบการมีอยู่ของ PyTorch
try:
    import importlib.util
    torch_spec = importlib.util.find_spec("torch")
    TORCH_AVAILABLE = torch_spec is not None
except ImportError:
    TORCH_AVAILABLE = False

class AIPredictor:
    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        self.model = None
        self.is_loaded = False
        
        # กฎเหล็กพารามิเตอร์ของโปรแกรม (ตามตัวไฟล์ predict_classifier.py)
        self.slice_height = 256
        self.target_width = 720
        self.stride = 32
        self.min_cut_distance = 1000
        self.safe_threshold = 0.85
        
        self.test_transforms = None

    def _init_torch(self):
        """โหลด PyTorch เฉพาะตอนที่ใช้งานจริงๆ เพื่อป้องกัน App Crash ตอนเปิด"""
        if not TORCH_AVAILABLE:
            return False
            
        try:
            global torch, nn, transforms, models
            import torch
            import torch.nn as nn
            from torchvision import transforms, models
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.test_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.slice_height, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            return True
        except Exception as e:
            print(f"Error loading PyTorch: {e}")
            return False

    def load_model(self):
        """โหลดโมเดลเข้าในหน่วยความจำ ถือว่าทำแค่ครั้งเดียวต่อการใช้งานตัวตัดภาพ"""
        if not self._init_torch():
            return False, "ไม่สามารถโหลดไลบรารี PyTorch ได้ (DLL Error หรือไม่ได้ติดตั้ง)"
            
        if not os.path.exists(self.weights_path):
            return False, f"ไม่พบไฟล์โมเดลที่: {self.weights_path}"
            
        try:
            self.model = models.efficientnet_b0(weights=None)
            in_features = self.model.classifier[1].in_features
            # 2 Classes: 0_unsafe, 1_safe
            self.model.classifier[1] = nn.Linear(in_features, 2)
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            return True, "โหลดโมเดลสำเร็จ"
        except Exception as e:
            return False, f"โหลดโมเดลล้มเหลว: {str(e)}"
            
    def predict_cut_points(self, pil_image, progress_callback=None):
        """รับภาพรวม (PIL Image) แล้วคืนค่าพิกัดตัด (List[int]) ของเส้นแนวนอน"""
        if not self.is_loaded:
            success, msg = self.load_model()
            if not success:
                return False, [], msg
                
        try:
            # 1. แปลงภาพ PIL(RGB) เป็น OpenCV(BGR)
            open_cv_image = np.array(pil_image)
            if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:
                img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            elif len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 4:
                img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGR)
            else:
                img = open_cv_image.copy()
            
            # 2. ปรับความกว้างให้ตรงกับโมเดลต้องการ (720px)
            h, w = img.shape[:2]
            scale_ratio = self.target_width / w
            if w != self.target_width:
                img = cv2.resize(img, (self.target_width, int(h * scale_ratio)))
                
            total_h = img.shape[0]
            
            # 3. เตรียมการรูด Sliding Window
            y_starts = list(range(0, total_h - self.slice_height, self.stride))
            if not y_starts:
                return True, [], "ภาพสั้นเกินไป ไม่จำเป็นต้องหาจุดตัด"
                
            scores = []
            batch_size = 64
            batch_tensors = []
            batch_y = []
            
            total_steps = len(y_starts)
            step_count = 0
            
            # รูดแว่นขยายทีละ Batch ส่งการ์ดจอ
            with torch.no_grad():
                for start_y in y_starts:
                    window = img[start_y:start_y + self.slice_height, :]
                    
                    if window.shape[1] != self.target_width:
                        window = cv2.resize(window, (self.target_width, self.slice_height))
                        
                    window_t = self.test_transforms(window)
                    batch_tensors.append(window_t)
                    batch_y.append(start_y)
                    
                    step_count += 1
                    # อัพเดท Progress Bar 0-80% ระหว่างสแกนภาพ
                    if progress_callback and step_count % batch_size == 0:
                        progress_callback(int((step_count / total_steps) * 80))
                    
                    if len(batch_tensors) == batch_size or start_y == y_starts[-1]:
                        batch_stack = torch.stack(batch_tensors).to(self.device)
                        outputs = self.model(batch_stack)
                        probs = torch.softmax(outputs, dim=1)
                        
                        # ใช้ Class 1 = Safe Score
                        safe_probs = probs[:, 1].cpu().numpy()
                        
                        for i, p in enumerate(safe_probs):
                            scores.append((batch_y[i] + (self.slice_height // 2), p))
                            
                        batch_tensors = []
                        batch_y = []
                        
            if progress_callback:
                progress_callback(90)
                
            # 4. แปลงกราฟหาจุดตัดพีค
            y_coords = np.array([s[0] for s in scores])
            safe_scores = np.array([s[1] for s in scores])
            
            distance_in_array = self.min_cut_distance // self.stride
            peaks, _ = find_peaks(safe_scores, height=self.safe_threshold, distance=distance_in_array)
            
            cut_ys = y_coords[peaks]
            # cut_probs = safe_scores[peaks]
            
            # 5. แปลงพิกัดสเกล Y กลับสู่ภาพสเกล Original ตามที่โปรแกรมกำลังเปิด (ไม่ได้ขึ้นกับ target_width 720 อีกต่อไป)
            original_scale_ratio = pil_image.height / total_h
            original_cut_ys = [int(y * original_scale_ratio) for y in cut_ys]
            
            if progress_callback:
                progress_callback(100)
                
            return True, original_cut_ys, f"พบจุดตัด {len(original_cut_ys)} จุด โดย AI"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, [], f"การวิเคราะห์ล้มเหลว: {str(e)}"
