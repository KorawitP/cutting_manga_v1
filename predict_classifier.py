import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms, models
import re
from tqdm import tqdm
from scipy.signal import find_peaks

def imread_unicode(path):
    with open(path, "rb") as stream:
        bytes_data = bytearray(stream.read())
        numpyarray = np.asarray(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

def imwrite_unicode(path, img):
    is_success, im_buf_arr = cv2.imencode(".jpg", img)
    if is_success:
        im_buf_arr.tofile(path)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_model(weights_path, device):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    # 2 Classes: 0_unsafe, 1_safe
    model.classifier[1] = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- กฎเหล็กพารามิเตอร์ ---
    ep_path = r"C:\Users\computer\Downloads\yolo\20" # ชี้ไปที่เรื่องที่ 20 เหมือนเดิม
    weights_path = r"C:\Users\computer\Downloads\yolo\best_classifier.pth"
    output_path = r"C:\Users\computer\Downloads\yolo\classifier_result.jpg"

    slice_height = 256
    target_width = 720
    stride = 32 # รูดแว่นขยายทีละ 32 พิกเซล (ถ้าอยากให้ละเอียดจัดๆ ปรับเป็น 16 หรือ 8 แต่จะช้า)
    
    # เงื่อนไขการสับหลีก
    min_cut_distance = 1000 # ตัดครั้งต่อไปต้องห่างอย่างน้อย 1000 px
    safe_threshold = 0.85 # จุดนั้นต้องปลอดภัย (Safe Score) เกิน 85% ถึงจะยอมกรีด

    # 1. โหลดเข้าทรง
    print(f"Loading episode from: {ep_path}")
    images = sorted([f for f in os.listdir(ep_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))], key=natural_sort_key)
    limit = 25
    print(f"Concatenating first {limit} images...")
    
    imgs_data = []
    current_target_w = None
    for img_name in images[:limit]:
        img = imread_unicode(os.path.join(ep_path, img_name))
        if img is None: continue
        
        if current_target_w is None: 
            current_target_w = img.shape[1]
        
        h, w = img.shape[:2]
        if w != current_target_w:
            img = cv2.resize(img, (current_target_w, int(h * current_target_w / w)))
        imgs_data.append(img)

    full_img = np.vstack(imgs_data)
    total_h, total_w = full_img.shape[:2]
    print(f"Giant image generated: {total_w}x{total_h} pixels")

    # 2. โหลด AI
    print("Loading AI Brain...")
    model = load_model(weights_path, device)
    
    # ท่ามารตฐานสำหรับส่งรูปเข้า EfficientNet
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((slice_height, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. รูดแว่นขยาย (Sliding Window)
    print("Starting Sliding Window Scan...")
    y_starts = list(range(0, total_h - slice_height, stride))
    scores = []
    
    # เราจะคำนวณทีละ Batch (เอาหลายหน้าต่างส่งให้การ์ดจอรวดเดียว จะได้เร็วขึ้น 10 เท่า)
    batch_size = 64
    batch_tensors = []
    batch_y = []

    with torch.no_grad():
        for start_y in tqdm(y_starts, desc="Scanning image"):
            # ตัดภาพใส่แว่นขยาย
            window = full_img[start_y:start_y + slice_height, :]
            
            # ย่อกว้างนิดนึงตามที่โมเดลขอ
            if window.shape[1] != target_width:
                window = cv2.resize(window, (target_width, slice_height))
                
            window_t = test_transforms(window)
            batch_tensors.append(window_t)
            batch_y.append(start_y)
            
            # ถ้าครบ Batch หรือหมดหลอดแล้ว ให้ส่งการ์ดจอรัน
            if len(batch_tensors) == batch_size or start_y == y_starts[-1]:
                batch_stack = torch.stack(batch_tensors).to(device)
                outputs = model(batch_stack)
                # ใช้ Softmax เพื่อแปลงคะแนนดิบเป็น % (0.0 - 1.0)
                probs = torch.softmax(outputs, dim=1) 
                
                # โพย: Class 0 = Unsafe, Class 1 = Safe (เราสนใจคะแนนความ Safe ในช่องที่ 1)
                safe_probs = probs[:, 1].cpu().numpy()
                
                for i, p in enumerate(safe_probs):
                    scores.append((batch_y[i] + (slice_height // 2), p)) # เซฟพิกัด Y ตรงกลางแว่นขยาย พร้อมคะแนน
                    
                batch_tensors = []
                batch_y = []

    # 4. แปลงกราฟหาจุดพีค (Find Peaks)
    print("\nAnalyzing safe zones...")
    y_coords = np.array([s[0] for s in scores])
    safe_scores = np.array([s[1] for s in scores])
    
    # อัลกอริทึมหา "ยอดเขาแห่งความปลอดภัย" สไตล์ SciPy
    # height: ยอดต้องสูงกว่า 85%
    # distance: ยอดแต่ละอันต้องห่างกันอย่างน้อย 1500px / stride (เพราะ scores เก็บอาเรย์แบบโดนย่อ stride)
    distance_in_array = min_cut_distance // stride
    peaks, properties = find_peaks(safe_scores, height=safe_threshold, distance=distance_in_array)
    
    cut_ys = y_coords[peaks]
    cut_probs = safe_scores[peaks]
    
    print(f"🎉 AI Found {len(cut_ys)} optimal cut points!")

    # 5. วาดเส้นลงภาพ (Visualization)
    result_img = full_img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i in range(len(cut_ys)):
        y = int(cut_ys[i])
        prob = cut_probs[i]
        
        # วาดเส้นตัดสีเขียวเรืองแสง (เรามั่นใจมากว่าตัดได้!)
        cv2.line(result_img, (0, y), (total_w, y), (0, 255, 0), 4)
        
        # วาดกล่องข้อความอธิบาย
        text = f"Safe Cut: {prob*100:.1f}%"
        cv2.rectangle(result_img, (10, y - 50), (450, y - 10), (0, 255, 0), -1)
        cv2.putText(result_img, text, (20, y - 15), font, 1.2, (0, 0, 0), 3)

    imwrite_unicode(output_path, result_img)
    print(f"Mission Complete! Awesome result saved to: {output_path}")

if __name__ == "__main__":
    main()
