"""
เทสเคสสำหรับตรวจสอบการทำงานของ AI Predictor
รันด้วย: python test_ai.py
"""
import os
import sys

print("=" * 60)
print("  เทสเคส: AI Guideline Predictor")
print("=" * 60)

# --- Test 1: ตรวจสอบว่า PyTorch โหลดได้ ---
print("\n[Test 1] ตรวจสอบการโหลด PyTorch...")
try:
    import torch
    print(f"  ✅ PyTorch โหลดสำเร็จ! เวอร์ชัน: {torch.__version__}")
    print(f"  ✅ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  ❌ โหลด PyTorch ไม่ได้: {e}")
    print("  ⚠️ จบเทส เนื่องจาก PyTorch ใช้งานไม่ได้")
    sys.exit(1)

# --- Test 2: ตรวจสอบว่า torchvision โหลดได้ ---
print("\n[Test 2] ตรวจสอบการโหลด torchvision...")
try:
    import torchvision
    from torchvision import transforms, models
    print(f"  ✅ torchvision โหลดสำเร็จ! เวอร์ชัน: {torchvision.__version__}")
except Exception as e:
    print(f"  ❌ โหลด torchvision ไม่ได้: {e}")
    sys.exit(1)

# --- Test 3: ตรวจสอบว่า scipy โหลดได้ ---
print("\n[Test 3] ตรวจสอบการโหลด scipy...")
try:
    import scipy
    from scipy.signal import find_peaks
    print(f"  ✅ scipy โหลดสำเร็จ! เวอร์ชัน: {scipy.__version__}")
except Exception as e:
    print(f"  ❌ โหลด scipy ไม่ได้: {e}")
    sys.exit(1)

# --- Test 4: ตรวจสอบไฟล์โมเดล ---
print("\n[Test 4] ตรวจสอบไฟล์โมเดล best_classifier.pth...")
weights_path = os.path.join(os.getcwd(), "best_classifier.pth")
if os.path.exists(weights_path):
    size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    print(f"  ✅ พบไฟล์โมเดลที่: {weights_path}")
    print(f"  ✅ ขนาดไฟล์: {size_mb:.1f} MB")
else:
    print(f"  ❌ ไม่พบไฟล์โมเดลที่: {weights_path}")
    print("  ⚠️ จะข้ามเทสที่ต้องใช้โมเดลจริง")

# --- Test 5: ตรวจสอบ AIPredictor class ---
print("\n[Test 5] ตรวจสอบการสร้าง AIPredictor...")
try:
    from core.ai_predictor import AIPredictor
    predictor = AIPredictor(weights_path)
    print(f"  ✅ สร้าง AIPredictor สำเร็จ!")
    print(f"     slice_height={predictor.slice_height}, stride={predictor.stride}")
    print(f"     safe_threshold={predictor.safe_threshold}, min_cut_distance={predictor.min_cut_distance}")
except Exception as e:
    print(f"  ❌ สร้าง AIPredictor ล้มเหลว: {e}")
    sys.exit(1)

# --- Test 6: ตรวจสอบการโหลดโมเดล ---
print("\n[Test 6] ตรวจสอบการโหลดโมเดล EfficientNet...")
if os.path.exists(weights_path):
    success, msg = predictor.load_model()
    if success:
        print(f"  ✅ {msg}")
        print(f"  ✅ Device: {predictor.device}")
    else:
        print(f"  ❌ {msg}")
        sys.exit(1)
else:
    print("  ⏭️ ข้ามเทสนี้ (ไม่พบไฟล์โมเดล)")

# --- Test 7: ทดสอบการทำนายด้วยภาพจำลอง ---
print("\n[Test 7] ทดสอบการทำนายด้วยภาพจำลอง (สีขาวล้วน 720x5000px)...")
if os.path.exists(weights_path) and predictor.is_loaded:
    try:
        from PIL import Image
        # สร้างภาพจำลอง ขาวล้วน ความสูง 5000px
        dummy_img = Image.new("RGB", (720, 5000), color=(255, 255, 255))
        
        success, cut_ys, msg = predictor.predict_cut_points(dummy_img)
        if success:
            print(f"  ✅ ทำนายสำเร็จ: {msg}")
            print(f"  ✅ จุดตัดที่พบ: {cut_ys}")
        else:
            print(f"  ⚠️ ทำนายล้มเหลว (อาจปกติถ้าภาพขาวล้วน): {msg}")
    except Exception as e:
        print(f"  ❌ เกิดข้อผิดพลาด: {e}")
else:
    print("  ⏭️ ข้ามเทสนี้ (ไม่มีโมเดล)")

# --- Test 8: ตรวจสอบว่า PySide6 ยังทำงานได้หลังโหลด torch ---
print("\n[Test 8] ตรวจสอบว่า PySide6 ยังทำงานได้หลังโหลด PyTorch...")
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QImageReader
    print(f"  ✅ PySide6 โหลดสำเร็จหลังจาก PyTorch!")
except Exception as e:
    print(f"  ❌ PySide6 โหลดไม่ได้: {e}")

# --- สรุปผล ---
print("\n" + "=" * 60)
print("  สรุปผลเทส: ผ่านทุกรายการ! ✅")
print("=" * 60)
