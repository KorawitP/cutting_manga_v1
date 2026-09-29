import sys

# โหลด PyTorch ก่อน PySide6 เพื่อแก้ปัญหา DLL Conflict บน Windows
try:
    import torch
except Exception:
    pass

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImageReader
from ui.main_window import ManualImageSplitter
from ui.ui_components import ModernStyle

if __name__ == '__main__':
    # สำหรับ PySide6 High DPI scaling เปิดใช้งานโดยอัตโนมัติ
    app = QApplication(sys.argv)
    
    # ปิดลิมิตการจัดสรรหน่วยความจำสำหรับภาพ (เพื่อรองรับภาพขนาดใหญ่)
    QImageReader.setAllocationLimit(0)  # 0 = ไม่จำกัด
    
    # ตั้งค่าธีมให้กับแอพพลิเคชัน
    window = ManualImageSplitter()
    ModernStyle.apply_theme(app, window.dark_mode)
    
    window.show()
    sys.exit(app.exec()) 