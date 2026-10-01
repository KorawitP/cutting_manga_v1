# รายงานการ Refactor โปรแกรมแบ่งภาพมังฮวา

## สรุปภาพรวม

โปรแกรม `main.py` เป็นแอปพลิเคชัน PyQt5 สำหรับแบ่งภาพมังฮวาด้วยตนเอง ปัจจุบันมีขนาด **2,310 บรรทัด** ซึ่งถือว่าใหญ่มากสำหรับไฟล์เดียว การ refactor จะช่วยให้โค้ดมีโครงสร้างที่ดีขึ้น บำรุงรักษาง่ายขึ้น และพัฒนาต่อได้ง่ายขึ้น

## การวิเคราะห์โครงสร้างปัจจุบัน

### คลาสหลักที่พบ:
1. **ModernStyle** (บรรทัด 22-280) - จัดการธีมและสไตล์
2. **ModernButton** (บรรทัด 282-320) - ปุ่มที่มีเอฟเฟกต์
3. **DropArea** (บรรทัด 322-370) - พื้นที่ลากและวางไฟล์
4. **LineAction** (บรรทัด 372-395) - จัดการ undo/redo
5. **ManualImageSplitter** (บรรทัด 397-1,100) - คลาสหลักของแอปพลิเคชัน
6. **ManualSplitView** (บรรทัด 1,102-1,900) - การแสดงผลและจัดการเส้นตัด
7. **ManualSplitDialog** (บรรทัด 1,902-2,310) - หน้าต่างสำหรับตัดภาพ

## ปัญหาที่พบ

### 1. ไฟล์ขนาดใหญ่เกินไป
- ไฟล์เดียวมี 2,310 บรรทัด
- ผสมผสานหลายความรับผิดชอบในไฟล์เดียว
- ยากต่อการบำรุงรักษาและพัฒนา

### 2. การผสมผสานของ GUI และ Core Logic
- โค้ด UI และ business logic อยู่ในคลาสเดียวกัน
- การประมวลผลภาพผสมกับการแสดงผล
- ยากต่อการทดสอบแยกส่วน

### 3. การจัดการ Dependencies
- Import ทั้งหมดอยู่ที่ด้านบนไฟล์
- ไม่มีการแยกการใช้งาน dependencies ตามโมดูล

## แผนการ Refactor

### โครงสร้างโฟลเดอร์ที่แนะนำ:

```
manga_splitter/
├── main.py                 # Entry point
├── core/                   # Core business logic
│   ├── __init__.py
│   ├── image_processor.py  # การประมวลผลภาพ
│   ├── file_manager.py     # การจัดการไฟล์
│   └── settings.py         # การจัดการการตั้งค่า
├── gui/                    # GUI components
│   ├── __init__.py
│   ├── main_window.py      # หน้าต่างหลัก
│   ├── split_dialog.py     # หน้าต่างตัดภาพ
│   ├── split_view.py       # การแสดงผลภาพ
│   └── widgets/            # Custom widgets
│       ├── __init__.py
│       ├── modern_button.py
│       ├── drop_area.py
│       └── style_manager.py
├── utils/                  # Utilities
│   ├── __init__.py
│   ├── logger.py          # การจัดการ log
│   └── constants.py       # ค่าคงที่
└── requirements.txt
```

## การแยกส่วนที่แนะนำ

### 1. Core Module (`core/`)

#### `image_processor.py`
```python
class ImageProcessor:
    """จัดการการประมวลผลภาพ"""
    
    def merge_images(self, image_paths, progress_callback=None)
    def split_image(self, image, h_breakpoints, v_breakpoints, output_dir, episode_num)
    def resize_image(self, image, scale_factor)
    def validate_images(self, image_paths)
```

#### `file_manager.py`
```python
class FileManager:
    """จัดการการอ่าน/เขียนไฟล์"""
    
    def load_images(self, paths)
    def save_split_images(self, images, output_dir, episode_num)
    def create_output_directory(self, base_path)
    def get_supported_formats(self)
```

#### `settings.py`
```python
class SettingsManager:
    """จัดการการตั้งค่าแอปพลิเคชัน"""
    
    def load_settings(self)
    def save_settings(self, settings)
    def get_theme_settings(self)
    def get_image_quality_settings(self)
```

### 2. GUI Module (`gui/`)

#### `main_window.py`
```python
class MainWindow(QMainWindow):
    """หน้าต่างหลักของแอปพลิเคชัน"""
    
    def __init__(self)
    def setup_ui(self)
    def handle_file_selection(self)
    def start_splitting_process(self)
```

#### `split_dialog.py`
```python
class SplitDialog(QDialog):
    """หน้าต่างสำหรับตัดภาพ"""
    
    def __init__(self, pixmap)
    def setup_ui(self)
    def get_breakpoints(self)
    def apply_settings(self)
```

#### `split_view.py`
```python
class SplitView(QGraphicsView):
    """การแสดงผลและจัดการเส้นตัด"""
    
    def __init__(self)
    def add_line(self, position, line_type)
    def handle_mouse_events(self)
    def manage_undo_redo(self)
```

### 3. Widgets Module (`gui/widgets/`)

#### `style_manager.py`
```python
class StyleManager:
    """จัดการธีมและสไตล์"""
    
    def apply_theme(self, app, is_dark=False)
    def get_theme_colors(self, theme_name)
    def create_stylesheet(self, theme)
```

#### `modern_button.py`
```python
class ModernButton(QPushButton):
    """ปุ่มที่มีเอฟเฟกต์สวยงาม"""
```

#### `drop_area.py`
```python
class DropArea(QLabel):
    """พื้นที่ลากและวางไฟล์"""
```

### 4. Utils Module (`utils/`)

#### `logger.py`
```python
class Logger:
    """จัดการการบันทึก log"""
    
    def setup_logger(self, log_dir)
    def log_message(self, message, level='INFO')
    def create_log_file(self)
```

#### `constants.py`
```python
# ค่าคงที่ต่างๆ
SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp']
DEFAULT_THEME = 'light'
MAX_IMAGE_SIZE = None
```

## ประโยชน์ของการ Refactor

### 1. การแยกความรับผิดชอบ (Separation of Concerns)
- **Core Logic** แยกจาก **GUI Logic**
- **Image Processing** แยกจาก **File Management**
- **Settings** แยกจาก **UI Components**

### 2. ความสามารถในการทดสอบ (Testability)
- สามารถทดสอบ core logic แยกจาก GUI
- Mock dependencies ได้ง่าย
- Unit test แต่ละโมดูลแยกกัน

### 3. การบำรุงรักษา (Maintainability)
- ไฟล์ขนาดเล็กลง อ่านง่ายขึ้น
- แก้ไขส่วนใดส่วนหนึ่งไม่กระทบส่วนอื่น
- เพิ่มฟีเจอร์ใหม่ได้ง่าย

### 4. การขยายตัว (Scalability)
- เพิ่มธีมใหม่ได้ง่าย
- เพิ่มรูปแบบไฟล์ใหม่ได้ง่าย
- เพิ่มฟีเจอร์การประมวลผลภาพใหม่ได้

### 5. การนำกลับมาใช้ (Reusability)
- Core components สามารถใช้ในโปรเจกต์อื่นได้
- Custom widgets สามารถนำไปใช้ที่อื่นได้
- Utility functions สามารถแชร์ได้

## ขั้นตอนการ Refactor

### Phase 1: เตรียมโครงสร้าง
1. สร้างโฟลเดอร์และไฟล์ `__init__.py`
2. ย้าย imports ไปยังโมดูลที่เหมาะสม
3. สร้าง constants และ utility functions

### Phase 2: แยก Core Logic
1. ย้าย image processing functions ไป `core/image_processor.py`
2. ย้าย file management ไป `core/file_manager.py`
3. ย้าย settings management ไป `core/settings.py`

### Phase 3: แยก GUI Components
1. ย้าย `ModernStyle` ไป `gui/widgets/style_manager.py`
2. ย้าย custom widgets ไป `gui/widgets/`
3. แยก main window และ dialogs

### Phase 4: Integration และ Testing
1. อัพเดท imports ในไฟล์ต่างๆ
2. ทดสอบการทำงานของแต่ละโมดูล
3. ทดสอบการทำงานรวมของระบบ

### Phase 5: Optimization
1. ปรับปรุง performance
2. เพิ่ม error handling
3. เพิ่ม documentation

## ตัวอย่างการใช้งานหลัง Refactor

```python
# main.py
from gui.main_window import MainWindow
from gui.widgets.style_manager import StyleManager
from core.settings import SettingsManager

def main():
    app = QApplication(sys.argv)
    
    # โหลดการตั้งค่า
    settings = SettingsManager()
    
    # ใช้ธีม
    style_manager = StyleManager()
    style_manager.apply_theme(app, settings.get_dark_mode())
    
    # สร้างหน้าต่างหลัก
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
```

## สรุป

การ refactor โปรแกรมนี้จะช่วยให้:

1. **โค้ดมีโครงสร้างที่ดีขึ้น** - แยกความรับผิดชอบชัดเจน
2. **พัฒนาต่อได้ง่ายขึ้น** - เพิ่มฟีเจอร์ใหม่ไม่กระทบส่วนเก่า
3. **บำรุงรักษาง่ายขึ้น** - แก้ไขปัญหาหรือปรับปรุงได้เฉพาะส่วน
4. **ทดสอบได้ง่ายขึ้น** - แยกทดสอบแต่ละส่วนได้
5. **ขยายตัวได้ดีขึ้น** - รองรับการเติบโตของโปรแกรมในอนาคต

การ refactor นี้จะใช้เวลาประมาณ 2-3 วัน แต่จะให้ประโยชน์ระยะยาวในการพัฒนาและบำรุงรักษาโปรแกรม