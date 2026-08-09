"""
ไฟล์สไตล์สำหรับโปรแกรมแบ่งภาพมังฮวา
แยกออกจาก main.py เพื่อลดขนาดไฟล์และจัดระเบียบโค้ด
"""

import re


def _lighten_color(color_str, factor):
    match = re.match(r"#([0-9a-fA-F]{6})", color_str)
    if match:
        hex_val = match.group(1)
        r, g, b = int(hex_val[0:2], 16), int(hex_val[2:4], 16), int(hex_val[4:6], 16)
        r = min(255, int(r + (255 - r) * (1 - factor)))
        g = min(255, int(g + (255 - g) * (1 - factor)))
        b = min(255, int(b + (255 - b) * (1 - factor)))
        return f"#{r:02x}{g:02x}{b:02x}"
    match = re.match(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", color_str)
    if match:
        r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
        r = min(255, int(r + (255 - r) * (1 - factor)))
        g = min(255, int(g + (255 - g) * (1 - factor)))
        b = min(255, int(b + (255 - b) * (1 - factor)))
        return f"rgb({r}, {g}, {b})"
    return color_str


def get_application_stylesheet(theme):
    """
    สร้างและคืนค่า stylesheet สำหรับแอพพลิเคชัน
    
    Args:
        theme (dict): ชุดสีธีมที่ต้องการใช้
        
    Returns:
        str: CSS stylesheet string
    """
    return f"""
        QMainWindow, QDialog {{
            background-color: {theme["background"]};
        }}
        
        QWidget {{
            font-family: 'Segoe UI', Tahoma, Arial;
            font-size: 10pt;
            color: {theme["text"]};
        }}
        
        QPushButton {{
            background-color: {theme["primary"]};
            color: {theme["card_bg"]};
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
            min-height: 30px;
        }}
        
        QPushButton:hover {{
            background-color: {theme["primary"] + "cc"};
        }}
        
        QPushButton:pressed {{
            background-color: {theme["primary"] + "99"};
        }}
        
        QPushButton:disabled {{
            background-color: {theme["border"]};
            color: {theme["text_light"]};
        }}
        
        QLabel {{
            color: {theme["text"]};
            padding: 2px;
        }}
        
        QGroupBox {{
            font-weight: bold;
            border: 1px solid {theme["border"]};
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 10px;
            background-color: {theme["card_bg"]};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: {theme["primary"]};
            background-color: transparent;
        }}
        
        QLineEdit, QComboBox {{
            background-color: {theme["card_bg"]};
            border: 1px solid {theme["border"]};
            border-radius: 5px;
            padding: 5px;
            min-height: 25px;
            selection-background-color: {theme["primary"]};
        }}
        
        QLineEdit:hover, QComboBox:hover {{
            border: 1px solid {theme["primary"]};
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        
        QProgressBar {{
            border: 1px solid {theme["border"]};
            border-radius: 5px;
            text-align: center;
            color: {theme["text"]};
            background-color: {theme["card_bg"]};
            height: 15px;
        }}
        
        QProgressBar::chunk {{
            background-color: {theme["secondary"]};
            width: 10px;
            border-radius: 2px;
        }}
        
        QTabWidget::pane {{
            border: 1px solid {theme["border"]};
            border-radius: 5px;
            background-color: {theme["card_bg"]};
        }}
        
        QTabBar::tab {{
            background-color: {theme["background"]};
            color: {theme["text"]};
            border: 1px solid {theme["border"]};
            border-bottom: none;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            padding: 6px 12px;
            margin-right: 2px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {theme["primary"]};
            color: {theme["card_bg"]};
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {theme["hover"]};
        }}
        
        QSlider::groove:horizontal {{
            border: none;
            height: 8px;
            background-color: {theme["border"]};
            border-radius: 4px;
        }}
        
        QSlider::handle:horizontal {{
            background-color: {theme["primary"]};
            border: none;
            width: 18px;
            height: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }}
        
        QSlider::handle:horizontal:hover {{
            background-color: {theme["accent"]};
        }}
        
        QStatusBar {{
            background-color: {theme["card_bg"]};
            color: {theme["text_light"]};
            border-top: 1px solid {theme["border"]};
        }}
        
        QSplitter::handle {{
            background-color: {theme["border"]};
        }}
        
        QSplitter::handle:horizontal {{
            width: 1px;
        }}
        
        QSplitter::handle:vertical {{
            height: 1px;
        }}
    """

def get_modern_button_style(color):
    """
    สร้างสไตล์สำหรับปุ่ม ModernButton
    
    Args:
        color (str): สีของปุ่ม
        
    Returns:
        str: CSS style string สำหรับปุ่ม
    """
    return f"""
        QPushButton {{
            background-color: {color};
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {color}cc;
        }}
        QPushButton:pressed {{
            background-color: {color}99;
        }}
    """

def get_splitter_style():
    """
    สร้างสไตล์สำหรับ QSplitter
    
    Returns:
        str: CSS style string สำหรับ splitter
    """
    return """
        QSplitter::handle {
            background-color: #dddddd;
        }
        QSplitter::handle:hover {
            background-color: #2196F3;
        }
    """

def get_tab_widget_style():
    """
    สร้างสไตล์สำหรับ QTabWidget
    
    Returns:
        str: CSS style string สำหรับ tab widget
    """
    return """
        QTabWidget::pane {
            border: 1px solid #dddddd;
            border-radius: 3px;
            background-color: #ffffff;
        }
        QTabWidget::tab-bar {
            left: 5px;
        }
        QTabBar::tab {
            background-color: #e0e0e0;
            color: #444444;
            padding: 6px 15px;
            border-top-left-radius: 3px;
            border-top-right-radius: 3px;
            margin-right: 2px;
            margin-bottom: -1px;
        }
        QTabBar::tab:selected {
            background-color: #ffffff;
            border-bottom: 2px solid #2196F3;
            color: #2196F3;
            font-weight: bold;
        }
        QTabBar::tab:hover:!selected {
            background-color: #eeeeee;
        }
    """

def get_slider_style():
    """
    สร้างสไตล์สำหรับ QSlider
    
    Returns:
        str: CSS style string สำหรับ slider
    """
    return """
        QSlider::groove:horizontal {
            height: 4px;
            background: #ddd;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #2196F3;
            width: 12px;
            margin: -4px 0;
            border-radius: 6px;
        }
        QSlider::handle:horizontal:hover {
            background: #1976D2;
        }
    """

def get_button_style(bg_color, hover_color=None, pressed_color=None):
    """
    สร้างสไตล์สำหรับปุ่มทั่วไป
    
    Args:
        bg_color (str): สีพื้นหลัง
        hover_color (str, optional): สีเมื่อ hover
        pressed_color (str, optional): สีเมื่อกด
        
    Returns:
        str: CSS style string สำหรับปุ่ม
    """
    hover = hover_color or _lighten_color(bg_color, 0.8)
    pressed = pressed_color or _lighten_color(bg_color, 0.6)
    
    return f"""
        QPushButton {{
            background-color: {bg_color};
            color: white;
            padding: 3px 10px;
            border: none;
            border-radius: 3px;
        }}
        QPushButton:hover {{ 
            background-color: {hover};
        }}
        QPushButton:pressed {{ 
            background-color: {pressed};
        }}
    """

def get_undo_redo_button_style():
    """
    สร้างสไตล์สำหรับปุ่ม Undo/Redo
    
    Returns:
        str: CSS style string สำหรับปุ่ม undo/redo
    """
    return """
        QPushButton {
            background-color: #607D8B;
            color: white;
            border: none;
            border-radius: 3px;
            font-size: 16px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #546E7A; }
        QPushButton:pressed { background-color: #455A64; }
        QPushButton:disabled { background-color: #B0BEC5; }
    """

def get_color_button_style():
    """
    สร้างสไตล์สำหรับปุ่มเลือกสี
    
    Returns:
        str: CSS style string สำหรับปุ่มสี
    """
    return """
        QPushButton {
            background-color: #f0f0f0;
            color: #444444;
            padding: 3px 8px;
            border: 1px solid #dddddd;
            border-radius: 3px;
        }
        QPushButton:hover { 
            background-color: #e0e0e0;
            border: 1px solid #bbbbbb;
        }
        QPushButton:pressed { background-color: #d0d0d0; }
    """