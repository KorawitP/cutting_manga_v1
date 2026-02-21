from PySide6.QtWidgets import QPushButton
from PySide6.QtGui import QPalette, QColor, QCursor
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QSize
from ui.styles import get_application_stylesheet, get_modern_button_style

class ModernStyle:
    """คลาสที่มีสไตล์และธีมต่างๆ สำหรับแอพพลิเคชัน"""
    
    # ชุดสีหลัก
    LIGHT_THEME = {
        "background": "#f5f5f7",
        "card_bg": "#ffffff",
        "primary": "#2196F3",
        "secondary": "#4CAF50",
        "accent": "#FF9800",
        "text": "#333333",
        "text_light": "#666666",
        "border": "#dddddd",
        "hover": "#e6f4ff",
        "error": "#f44336",
        "success": "#4caf50",
        "warning": "#ff9800",
    }
    
    DARK_THEME = {
        "background": "#1a1a1a",
        "card_bg": "#2d2d2d",
        "primary": "#1e88e5",
        "secondary": "#43a047",
        "accent": "#ff9100",
        "text": "#f0f0f0",
        "text_light": "#b0b0b0",
        "border": "#444444",
        "hover": "#333f50",
        "error": "#ef5350",
        "success": "#66bb6a",
        "warning": "#ffb74d",
    }
    
    @staticmethod
    def get_theme(is_dark=False):
        """รับชุดธีมที่เลือก"""
        return ModernStyle.DARK_THEME if is_dark else ModernStyle.LIGHT_THEME
    
    @staticmethod
    def apply_theme(app, is_dark=False):
        """ใช้ธีมกับแอพพลิเคชัน"""
        theme = ModernStyle.get_theme(is_dark)
        
        # ตั้งค่าสไตล์ทั่วไป
        app.setStyle("Fusion")
        
        # กำหนดชุดสีสำหรับแอพพลิเคชันทั้งหมด
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(theme["background"]))
        palette.setColor(QPalette.WindowText, QColor(theme["text"]))
        palette.setColor(QPalette.Base, QColor(theme["card_bg"]))
        palette.setColor(QPalette.AlternateBase, QColor(theme["hover"]))
        palette.setColor(QPalette.ToolTipBase, QColor(theme["card_bg"]))
        palette.setColor(QPalette.ToolTipText, QColor(theme["text"]))
        palette.setColor(QPalette.Text, QColor(theme["text"]))
        palette.setColor(QPalette.Button, QColor(theme["card_bg"]))
        palette.setColor(QPalette.ButtonText, QColor(theme["text"]))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(theme["primary"]))
        palette.setColor(QPalette.Highlight, QColor(theme["primary"]))
        palette.setColor(QPalette.HighlightedText, QColor(theme["card_bg"]))
        
        # ตั้งค่าแต่ละกลุ่มสี
        for group in [QPalette.Active, QPalette.Inactive, QPalette.Disabled]:
            palette.setColor(group, QPalette.WindowText, QColor(theme["text"]))
            palette.setColor(group, QPalette.Text, QColor(theme["text"]))
            palette.setColor(group, QPalette.ButtonText, QColor(theme["text"]))
            if group == QPalette.Disabled:
                palette.setColor(group, QPalette.WindowText, QColor(theme["text_light"]))
                palette.setColor(group, QPalette.Text, QColor(theme["text_light"]))
                palette.setColor(group, QPalette.ButtonText, QColor(theme["text_light"]))
        
        # ใช้ชุดสีกับแอพพลิเคชัน
        app.setPalette(palette)
        
        # ใช้ stylesheet จากไฟล์ styles.py
        stylesheet = get_application_stylesheet(theme)
        app.setStyleSheet(stylesheet)

class ModernButton(QPushButton):
    """ปุ่มโมเดิร์นที่มีเอฟเฟกต์เมื่อวางเมาส์เหนือและคลิก"""
    
    def __init__(self, text="", parent=None, icon=None, color=None):
        super().__init__(text, parent)
        self._animation = QPropertyAnimation(self, b"pos")
        self._animation.setDuration(100)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        
        if icon:
            self.setIcon(icon)
            self.setIconSize(QSize(20, 20))
        
        if color:
            self.setStyleSheet(get_modern_button_style(color))
    
    def enterEvent(self, event):
        self.setCursor(QCursor(Qt.PointingHandCursor))
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        self.setCursor(QCursor(Qt.ArrowCursor))
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.pos()
            self._animation.setStartValue(pos)
            self._animation.setEndValue(pos)
            self._animation.start()
        super().mousePressEvent(event)
