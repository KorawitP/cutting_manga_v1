import os
import sys
import json
import random
import shutil
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QFileDialog, QLabel, QMessageBox,
                           QHBoxLayout, QDialog, QScrollArea, QProgressDialog,
                           QProgressBar, QGroupBox, QSpinBox, QInputDialog, QLineEdit,
                           QSlider, QComboBox, QCheckBox, QTabWidget, QFrame, QStatusBar,
                           QSplitter, QToolButton, QStyle, QColorDialog)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QIcon, QCursor, QPalette, QBrush, QImageReader
from PySide6.QtCore import Qt, QPointF, QRectF, QPropertyAnimation, QEasingCurve, QSize, QSettings, Signal
from PIL import Image, ImageEnhance
import numpy as np
from datetime import datetime
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsLineItem
from io import BytesIO
import gc
import time

# ปิดคำเตือน decompression bomb สำหรับภาพขนาดใหญ่
Image.MAX_IMAGE_PIXELS = None

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
        
        # สร้างสไตล์ชีทสำหรับองค์ประกอบต่างๆ
        stylesheet = f"""
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
            
            QSpinBox, QLineEdit, QComboBox {{
                background-color: {theme["card_bg"]};
                border: 1px solid {theme["border"]};
                border-radius: 5px;
                padding: 5px;
                min-height: 25px;
                selection-background-color: {theme["primary"]};
            }}
            
            QSpinBox:hover, QLineEdit:hover, QComboBox:hover {{
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
            
            QScrollBar:vertical {{
                border: none;
                background-color: {theme["background"]};
                width: 12px;
                margin: 0px;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: {theme["border"]};
                min-height: 30px;
                border-radius: 6px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background-color: {theme["primary"]};
            }}
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
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
            
            QCheckBox {{
                spacing: 5px;
            }}
            
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 1px solid {theme["border"]};
                border-radius: 3px;
                background-color: {theme["card_bg"]};
            }}
            
            QCheckBox::indicator:checked {{
                background-color: {theme["primary"]};
                border: none;
                image: url(:/images/check.png);
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
            
            QToolTip {{
                background-color: {theme["card_bg"]};
                color: {theme["text"]};
                border: 1px solid {theme["border"]};
                border-radius: 4px;
                padding: 5px;
                opacity: 200;
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
            self.setStyleSheet(f"""
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
            """)
    
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

class LineAction:
    """คลาสสำหรับเก็บข้อมูลการกระทำเพื่อใช้กับ undo/redo"""
    
    ADD = "add"        # เพิ่มเส้น
    REMOVE = "remove"  # ลบเส้น
    MOVE = "move"      # ย้ายเส้น
    
    # เพิ่มตัวแปรคงที่เก็บประเภทเส้น
    HORIZONTAL = "horizontal"  # เส้นแนวนอน
    VERTICAL = "vertical"      # เส้นแนวตั้ง
    
    def __init__(self, action_type, line_index=None, line_data=None, old_pos=None, new_pos=None):
        self.type = action_type            # ประเภทการกระทำ
        self.line_index = line_index       # ตำแหน่งของเส้นในรายการ
        self.line_data = line_data         # ข้อมูลเส้น (สำหรับเพิ่ม/ลบ)
        self.old_pos = old_pos             # ตำแหน่งเก่า (สำหรับย้าย) - เป็น y สำหรับเส้นแนวนอน หรือ x สำหรับเส้นแนวตั้ง
        self.new_pos = new_pos             # ตำแหน่งใหม่ (สำหรับย้าย) - เป็น y สำหรับเส้นแนวนอน หรือ x สำหรับเส้นแนวตั้ง

class ManualImageSplitter(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # โหลดการตั้งค่า
        self.settings = QSettings("ManualImageSplitter", "Settings")
        self.dark_mode = self.settings.value("dark_mode", False, type=bool)
        
        # ตั้งค่าธีม
        self.theme = ModernStyle.get_theme(self.dark_mode)
        
        self.setWindowTitle("โปรแกรมแบ่งภาพมังฮวาด้วยตนเอง")
        self.setGeometry(100, 100, 900, 700)
        
        # ตั้งค่าไอคอน (ถ้ามี)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        
        # เพิ่มการกำหนดไฟล์ log
        self.log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"manual_splitter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        # ตัวแปรเก็บไฟล์
        self.image_files = []
        
        # ตั้งค่าเริ่มต้นสำหรับการบันทึก
        self.default_output_path = ""
        self.default_episode_name = ""
        
        # เริ่มต้นสร้าง UI
        self.setup_ui()
        
        # เพิ่ม status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("พร้อมใช้งาน")
        
        self.log_message("เริ่มต้นโปรแกรม")
    
    def setup_ui(self):
        """ตั้งค่าอินเตอร์เฟซผู้ใช้"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # ส่วนหัว
        header_layout = QHBoxLayout()
        
        app_title = QLabel("โปรแกรมแบ่งภาพมังฮวาด้วยตนเอง")
        app_title.setStyleSheet(f"""
            font-size: 18px;
            font-weight: bold;
            color: {self.theme["primary"]};
        """)
        header_layout.addWidget(app_title)
        
        header_layout.addStretch()
        
        # ปุ่มธีม
        self.theme_btn = QToolButton()
        self.theme_btn.setIcon(self.style().standardIcon(
            QStyle.SP_DialogHelpButton if self.dark_mode else QStyle.SP_DialogCloseButton
        ))
        self.theme_btn.setIconSize(QSize(24, 24))
        self.theme_btn.setToolTip("สลับธีมมืด/สว่าง")
        self.theme_btn.clicked.connect(self.toggle_theme)
        header_layout.addWidget(self.theme_btn)
        
        main_layout.addLayout(header_layout)
        
        # แท็บสำหรับแบ่งหน้าที่
        tab_widget = QTabWidget()
        
        # แท็บหลัก
        main_tab = QWidget()
        main_tab_layout = QVBoxLayout(main_tab)
        
        # ปุ่มควบคุม
        button_layout = QHBoxLayout()
        
        self.select_btn = ModernButton("เลือกไฟล์ภาพ", self, 
                                     self.style().standardIcon(QStyle.SP_FileDialogStart),
                                     self.theme["primary"])
        self.select_btn.clicked.connect(self.select_files)
        button_layout.addWidget(self.select_btn)
        
        self.reset_btn = ModernButton("รีเซตรายการ", self,
                                    self.style().standardIcon(QStyle.SP_DialogResetButton),
                                    self.theme["error"])
        self.reset_btn.clicked.connect(self.reset_files)
        button_layout.addWidget(self.reset_btn)
        
        main_tab_layout.addLayout(button_layout)
        
        # แสดงจำนวนไฟล์ที่เลือก
        file_info_group = QGroupBox("รายละเอียดไฟล์")
        file_info_layout = QVBoxLayout(file_info_group)
        
        self.file_label = QLabel("ยังไม่ได้เลือกไฟล์")
        self.file_label.setAlignment(Qt.AlignCenter)
        file_info_layout.addWidget(self.file_label)
        
        main_tab_layout.addWidget(file_info_group)
        
        # เพิ่มส่วนการตั้งค่าการบันทึก
        save_settings_group = QGroupBox("การตั้งค่าการบันทึก")
        save_settings_layout = QVBoxLayout(save_settings_group)
        
        # ช่องใส่พาธปลายทาง
        output_path_layout = QHBoxLayout()
        output_path_label = QLabel("โฟลเดอร์ปลายทาง:")
        output_path_layout.addWidget(output_path_label)
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("เลือกโฟลเดอร์สำหรับบันทึกภาพที่ตัดแล้ว...")
        output_path_layout.addWidget(self.output_path_edit)
        
        self.browse_btn = ModernButton("...", self,
                                     self.style().standardIcon(QStyle.SP_DirOpenIcon),
                                     self.theme["accent"])
        self.browse_btn.setMaximumWidth(40)
        self.browse_btn.clicked.connect(self.browse_output_folder)
        output_path_layout.addWidget(self.browse_btn)
        
        save_settings_layout.addLayout(output_path_layout)
        
        # ช่องใส่เลขตอน/ชื่อไฟล์
        episode_layout = QHBoxLayout()
        episode_label = QLabel("เลขตอน/ชื่อไฟล์:")
        episode_layout.addWidget(episode_label)
        
        self.episode_edit = QLineEdit()
        self.episode_edit.setPlaceholderText("เช่น 10 จะได้ 10_1, 10_2, ...")
        episode_layout.addWidget(self.episode_edit)
        
        save_settings_layout.addLayout(episode_layout)
        
        main_tab_layout.addWidget(save_settings_group)
        
        # Progress Bar
        progress_group = QGroupBox("ความคืบหน้า")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_label = QLabel("กำลังรอดำเนินการ...")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        main_tab_layout.addWidget(progress_group)
        
        # ปุ่มเริ่มแบ่งภาพ
        start_layout = QHBoxLayout()
        start_layout.addStretch()
        
        self.split_btn = ModernButton("เริ่มแบ่งภาพ", self,
                                    self.style().standardIcon(QStyle.SP_MediaPlay),
                                    self.theme["secondary"])
        self.split_btn.setMinimumWidth(200)
        self.split_btn.setMinimumHeight(40)
        self.split_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme["secondary"]};
                color: white;
                border: none;
                border-radius: 20px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.theme["secondary"] + "dd"};
            }}
            QPushButton:pressed {{
                background-color: {self.theme["secondary"] + "aa"};
            }}
        """)
        self.split_btn.clicked.connect(self.start_manual_split)
        start_layout.addWidget(self.split_btn)
        
        start_layout.addStretch()
        main_tab_layout.addLayout(start_layout)
        
        # เพิ่มสเปซเพื่อให้มีพื้นที่ว่างด้านล่าง
        main_tab_layout.addStretch()
        
        # แท็บการตั้งค่า
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        image_group = QGroupBox("การจัดการภาพ")
        image_layout = QVBoxLayout(image_group)
        
        # ตั้งค่าความละเอียดการบันทึกภาพ
        quality_layout = QHBoxLayout()
        quality_label = QLabel("คุณภาพการบันทึก:")
        quality_layout.addWidget(quality_label)
        
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setMinimum(30)
        self.quality_slider.setMaximum(100)
        self.quality_slider.setValue(self.settings.value("image_quality", 90, type=int))
        self.quality_slider.setTickPosition(QSlider.TicksBelow)
        self.quality_slider.setTickInterval(10)
        quality_layout.addWidget(self.quality_slider)
        
        self.quality_value = QLabel(f"{self.quality_slider.value()}%")
        quality_layout.addWidget(self.quality_value)
        
        self.quality_slider.valueChanged.connect(lambda v: self.quality_value.setText(f"{v}%"))
        self.quality_slider.sliderReleased.connect(lambda: self.settings.setValue("image_quality", self.quality_slider.value()))
        
        image_layout.addLayout(quality_layout)
        
        # รูปแบบการบันทึกไฟล์
        format_layout = QHBoxLayout()
        format_label = QLabel("รูปแบบไฟล์:")
        format_layout.addWidget(format_label)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPEG", "WebP"])
        current_format = self.settings.value("image_format", "PNG")
        self.format_combo.setCurrentText(current_format)
        self.format_combo.currentTextChanged.connect(lambda t: self.settings.setValue("image_format", t))
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        
        image_layout.addLayout(format_layout)
        
        settings_layout.addWidget(image_group)
        settings_layout.addStretch()
        
        # เพิ่มแท็บเข้าไป
        tab_widget.addTab(main_tab, "หน้าหลัก")
        tab_widget.addTab(settings_tab, "ตั้งค่า")
        
        main_layout.addWidget(tab_widget)
    
    def toggle_theme(self):
        """สลับระหว่างธีมมืดและสว่าง"""
        self.dark_mode = not self.dark_mode
        # แจ้งเตือนผู้ใช้ให้รีสตาร์ทแอพพลิเคชัน
        QMessageBox.information(self, "เปลี่ยนธีม", 
                              "การเปลี่ยนธีมจะมีผลเมื่อรีสตาร์ทโปรแกรม")

    def select_files(self):
        """เลือกไฟล์ภาพผ่าน File Dialog"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "เลือกไฟล์ภาพ",
            "",
            "Image Files (*.png *.jpg *.jpeg *.webp)"
        )
        
        if files:
            self.image_files.extend(files)
            self.update_file_label()
            self.log_message(f"เลือกไฟล์: {len(files)} ไฟล์")
            self.statusBar.showMessage(f"เลือก {len(files)} ไฟล์สำเร็จ", 3000)
            
            # ตั้งค่าเริ่มต้นสำหรับโฟลเดอร์ปลายทาง
            if not self.output_path_edit.text():
                default_folder = os.path.dirname(files[0])
                self.output_path_edit.setText(default_folder)
                self.log_message(f"ตั้งค่าโฟลเดอร์ปลายทางเริ่มต้น: {default_folder}")

    def reset_files(self):
        """ล้างรายการไฟล์"""
        if not self.image_files:
            return
            
        confirm = QMessageBox.question(
            self,
            "ยืนยันการรีเซต",
            "คุณต้องการล้างรายการไฟล์ทั้งหมดใช่หรือไม่?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            self.image_files = []
            self.update_file_label()
            # ไม่ล้างช่องข้อมูลการบันทึก - เก็บไว้ให้ผู้ใช้
            self.log_message("รีเซตรายการไฟล์")
            self.statusBar.showMessage("รีเซตรายการไฟล์แล้ว", 3000)

    def update_file_label(self):
        """อัพเดทข้อความแสดงจำนวนไฟล์"""
        if self.image_files:
            # แสดงจำนวนไฟล์และขนาดรวม
            total_size = 0
            for file in self.image_files:
                try:
                    total_size += os.path.getsize(file)
                except:
                    pass
            
            # แปลงขนาดให้อ่านง่าย
            if total_size < 1024:
                size_str = f"{total_size} bytes"
            elif total_size < 1024 * 1024:
                size_str = f"{total_size / 1024:.1f} KB"
            elif total_size < 1024 * 1024 * 1024:
                size_str = f"{total_size / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{total_size / (1024 * 1024 * 1024):.1f} GB"
            
            self.file_label.setText(
                f"ไฟล์ที่เลือก: {len(self.image_files)} ไฟล์\n"
                f"ขนาดรวม: {size_str}\n"
                f"ไฟล์แรก: {os.path.basename(self.image_files[0])}"
            )
        else:
            self.file_label.setText("ยังไม่ได้เลือกไฟล์")

    def browse_output_folder(self):
        """เลือกโฟลเดอร์ปลายทางสำหรับบันทึกไฟล์"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "เลือกโฟลเดอร์สำหรับบันทึกภาพที่ตัดแล้ว",
            self.output_path_edit.text() or (os.path.dirname(self.image_files[0]) if self.image_files else "")
        )
        
        if folder:
            self.output_path_edit.setText(folder)
            self.log_message(f"เลือกโฟลเดอร์ปลายทาง: {folder}")

    def validate_save_settings(self):
        """ตรวจสอบความถูกต้องของการตั้งค่าการบันทึก"""
        output_path = self.output_path_edit.text().strip()
        episode_name = self.episode_edit.text().strip()
        
        # ตรวจสอบโฟลเดอร์ปลายทาง
        if not output_path:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเลือกโฟลเดอร์ปลายทางสำหรับบันทึกไฟล์")
            return False
            
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path, exist_ok=True)
                self.log_message(f"สร้างโฟลเดอร์: {output_path}")
            except Exception as e:
                QMessageBox.critical(self, "ผิดพลาด", f"ไม่สามารถสร้างโฟลเดอร์ได้: {str(e)}")
                return False
        
        # ตรวจสอบสิทธิ์การเขียนไฟล์
        if not os.access(output_path, os.W_OK):
            QMessageBox.critical(self, "ผิดพลาด", "ไม่มีสิทธิ์เขียนไฟล์ในโฟลเดอร์ที่เลือก")
            return False
        
        # ตรวจสอบชื่อไฟล์
        if not episode_name:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณากรอกเลขตอนหรือชื่อไฟล์")
            return False
            
        # ตรวจสอบอักขระที่ไม่ถูกต้องในชื่อไฟล์
        invalid_chars = '<>:"/\\|?*'
        if any(char in episode_name for char in invalid_chars):
            QMessageBox.warning(self, "แจ้งเตือน", 
                              f"ชื่อไฟล์ไม่สามารถมีอักขระเหล่านี้: {invalid_chars}")
            return False
        
        return True

    def log_message(self, message):
        """บันทึกข้อความลงในไฟล์ log และแสดงที่หน้าจอ"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message_with_time = f"[{timestamp}] {message}"
        
        # บันทึกลงไฟล์
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message_with_time + '\n')
        
        print(message_with_time)

    def start_manual_split(self):
        """เริ่มกระบวนการแบ่งภาพ"""
        if not self.image_files:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเลือกไฟล์ภาพก่อน")
            return
        
        # ตรวจสอบการตั้งค่าการบันทึกก่อนเริ่มตัด
        if not self.validate_save_settings():
            return
        
        # รับค่าจากช่องที่ผู้ใช้กรอก
        output_dir = self.output_path_edit.text().strip()
        episode_num = self.episode_edit.text().strip()
        
        try:
            # แสดง progress dialog
            progress = QProgressDialog("กำลังเตรียมภาพ...", "ยกเลิก", 0, len(self.image_files), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            
            # รวมภาพ
            self.log_message("\nกำลังรวมภาพทั้งหมดเข้าด้วยกัน...")
            merged_img = self.merge_all_images(progress)
            
            if merged_img is None or progress.wasCanceled():
                return
            
            # แปลงเป็น QPixmap
            progress.setLabelText("กำลังเตรียมภาพสำหรับแสดงผล...")
            QApplication.processEvents()  # อัพเดท UI
            
            # ใช้ BytesIO เพื่อประหยัดหน่วยความจำ
            with BytesIO() as bio:
                merged_img.save(bio, format='PNG')
                qimg = QImage.fromData(bio.getvalue())
                pixmap = QPixmap.fromImage(qimg)
            
            progress.close()
            
            # แสดงหน้าต่างแบ่งภาพ
            self.log_message("เปิดหน้าต่างแบ่งภาพด้วยตนเอง...")
            dialog = ManualSplitDialog(self, pixmap)
            if dialog.exec() == QDialog.Accepted:
                h_breakpoints = dialog.get_horizontal_breakpoints()
                v_breakpoints = dialog.get_vertical_breakpoints()
                scale_factor = dialog.get_scale_factor()
                
                if h_breakpoints or v_breakpoints:
                    success = self.split_at_breakpoints(merged_img, h_breakpoints, v_breakpoints, scale_factor, output_dir, episode_num)
                    if success:
                        QMessageBox.information(self, "สำเร็จ", "แบ่งภาพเสร็จสิ้น")
                        # รีเซ็ตรายการภาพหลังจากตัดเสร็จ
                        self.reset_files()
                        self.log_message("รีเซ็ตรายการภาพหลังจากตัดเสร็จ")
        
        except Exception as e:
            self.log_message(f"เกิดข้อผิดพลาด: {str(e)}")
            QMessageBox.critical(self, "ผิดพลาด", f"เกิดข้อผิดพลาด: {str(e)}")
            
        finally:
            # ทำความสะอาดหน่วยความจำ
            gc.collect()  # เรียก garbage collector

    def merge_all_images(self, progress):
        """รวมทุกภาพในแนวตั้งเป็นภาพเดียว"""
        try:
            # เปิดและรวมภาพทีละภาพ
            total_height = 0
            width = None
            images = []
            
            for i, img_path in enumerate(self.image_files):
                progress.setValue(i)
                QApplication.processEvents()  # อัพเดท UI
                
                if progress.wasCanceled():
                    return None
                
                try:
                    img = Image.open(img_path)
                    if width is None:
                        width = img.width
                    elif img.width != width:
                        raise Exception(f"ความกว้างของภาพไม่เท่ากัน: {img_path}")
                    total_height += img.height
                    images.append(img)
                except Exception as e:
                    self.log_message(f"ข้อผิดพลาดในการโหลดภาพ {img_path}: {str(e)}")
                    continue
            
            if not images:
                raise Exception("ไม่สามารถโหลดภาพได้")
            
            # สร้างภาพรวม
            progress.setLabelText("กำลังรวมภาพ...")
            merged_img = Image.new('RGB', (width, total_height), 'white')
            current_height = 0
            
            for i, img in enumerate(images):
                progress.setValue(i)
                QApplication.processEvents()  # อัพเดท UI
                
                merged_img.paste(img, (0, current_height))
                current_height += img.height
                img.close()  # ปิดภาพทันทีหลังใช้งาน
            
            self.log_message(f"รวมภาพสำเร็จ - ขนาดภาพรวม: {width}x{total_height} pixels")
            return merged_img
            
        except Exception as e:
            self.log_message(f"เกิดข้อผิดพลาดในการรวมภาพ: {str(e)}")
            return None

    def split_at_breakpoints(self, merged_img, h_breakpoints, v_breakpoints, scale_factor=1.0, output_dir=None, episode_num=None):
        """แบ่งภาพตามจุดตัดที่กำหนด"""
        try:
            # สร้างโฟลเดอร์เก็บผลลัพธ์
            if output_dir is None:
                base_folder = os.path.dirname(self.image_files[0])
                folder_name = os.path.splitext(os.path.basename(self.image_files[0]))[0]
                output_dir = os.path.join(base_folder, f"split_results_{folder_name}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # ย่อขนาดภาพถ้าจำเป็น
            if scale_factor != 1.0:
                new_width = int(merged_img.width * scale_factor)
                new_height = int(merged_img.height * scale_factor)
                merged_img = merged_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.log_message(f"ย่อขนาดภาพเป็น {scale_factor*100}% ({new_width}x{new_height})")
            
            # ตรวจสอบว่ามีการกำหนดจุดตัดหรือไม่
            if not h_breakpoints and not v_breakpoints:
                self.log_message("ไม่มีจุดตัดที่กำหนด ข้ามการแบ่งภาพ")
                return True
                
            # เริ่มต้นจากภาพต้นฉบับที่จะใช้ตัด
            working_image = merged_img
            
            # ตัดขอบซ้ายและขวาตามเส้นแนวตั้งก่อน (ถ้ามี)
            if v_breakpoints:
                self.log_message(f"จุดตัดแนวตั้ง: {v_breakpoints}")
                
                # ตรวจสอบและเรียงลำดับจุดตัดแนวตั้ง
                v_breakpoints = sorted(v_breakpoints)
                
                # หากมีการตั้งจุดตัดแนวตั้งเพียง 2 จุด คือตัดขอบซ้ายและขวา
                # จะใช้จุดเริ่มต้นและจุดสิ้นสุดเป็นขอบเขตการตัด
                left_edge = v_breakpoints[0]
                right_edge = v_breakpoints[-1] if len(v_breakpoints) > 1 else merged_img.width
                
                # ตัดภาพตามขอบซ้ายและขวา
                cropped_image = merged_img.crop((left_edge, 0, right_edge, merged_img.height))
                working_image = cropped_image  # ใช้ภาพที่ตัดขอบแล้วในขั้นตอนต่อไป
                
                self.log_message(f"ตัดขอบซ้ายที่ {left_edge}px และขอบขวาที่ {right_edge}px แล้ว")
            
            # บันทึกผลลัพธ์ภาพที่ตัดแล้ว
            result_images = []
            
            # เมื่อตัดขอบซ้าย-ขวาแล้ว ต่อไปตัดตามแนวนอน (ถ้ามี)
            if h_breakpoints:
                # เพิ่มค่า 0 ที่จุดเริ่มต้นและความสูงภาพที่จุดสุดท้าย
                h_points = [0] + h_breakpoints + [working_image.height]
                self.log_message(f"จุดตัดแนวนอน: {h_breakpoints}")
                
                # สร้าง progress dialog สำหรับการตัดแนวนอน
                h_progress = QProgressDialog("กำลังแบ่งภาพ...", "ยกเลิก", 0, len(h_points) - 1, self)
                h_progress.setWindowModality(Qt.WindowModal)
                h_progress.setMinimumDuration(0)
                
                for i in range(len(h_points) - 1):
                    h_progress.setValue(i)
                    h_progress.setLabelText(f"กำลังแบ่งภาพส่วนที่ {i+1}/{len(h_points)-1}...")
                    QApplication.processEvents()  # อัพเดท UI
                    
                    if h_progress.wasCanceled():
                        return False
                    
                    y_start = h_points[i]
                    y_end = h_points[i+1]
                    
                    # ตัดภาพตามพิกัด y
                    final_crop = working_image.crop((0, y_start, working_image.width, y_end))
                    
                    # บันทึกข้อมูลภาพที่ตัดแล้ว
                    result_images.append({
                        "image": final_crop,
                        "horz_idx": 1,  # เพราะเราตัดขอบซ้าย-ขวาเป็นภาพเดียวแล้ว
                        "vert_idx": i + 1,
                        "width": final_crop.width,
                        "height": final_crop.height
                    })
                
                h_progress.close()
            else:
                # ถ้าไม่มีการตัดแนวนอน ให้ใช้ภาพที่ตัดขอบซ้าย-ขวาแล้วทั้งภาพ
                result_images.append({
                    "image": working_image,
                    "horz_idx": 1,
                    "vert_idx": 1,
                    "width": working_image.width,
                    "height": working_image.height
                })
            
            # บันทึกไฟล์ทั้งหมด
            save_progress = QProgressDialog("กำลังบันทึกไฟล์...", "ยกเลิก", 0, len(result_images), self)
            save_progress.setWindowModality(Qt.WindowModal)
            save_progress.setMinimumDuration(0)
            
            processed_images = []
            
            for i, img_info in enumerate(result_images):
                save_progress.setValue(i)
                save_progress.setLabelText(f"กำลังบันทึกไฟล์ {i+1}/{len(result_images)}...")
                QApplication.processEvents()  # อัพเดท UI
                
                if save_progress.wasCanceled():
                    return False
                
                # สร้างชื่อไฟล์
                if episode_num:
                    if v_breakpoints:
                        filename = f"{episode_num}_{img_info['vert_idx']}.png"
                    else:
                        filename = f"{episode_num}_{img_info['vert_idx']}.png"
                else:
                    folder_name = os.path.splitext(os.path.basename(self.image_files[0]))[0]
                    if v_breakpoints:
                        filename = f"{folder_name}_{img_info['vert_idx']}.png"
                    else:
                        filename = f"{folder_name}_{img_info['vert_idx']}.png"
                
                filepath = os.path.join(output_dir, filename)
                img_info["image"].save(filepath, "PNG")
                
                # เก็บข้อมูลสำหรับใช้ในการสร้าง Dataset
                processed_images.append({
                    "filepath": filepath,
                    "width": img_info["width"],
                    "height": img_info["height"]
                })
                
                self.log_message(f"บันทึกภาพ: {filepath}")
            
            save_progress.close()
            
            # เสร็จสิ้น
            total_images = len(result_images)
            self.log_message(f"แบ่งภาพเสร็จสิ้น {total_images} ส่วน บันทึกที่: {output_dir}")
            
            return True
            
        except Exception as e:
            self.log_message(f"เกิดข้อผิดพลาดในการแบ่งภาพ: {str(e)}")
            return False

class ManualSplitView(QGraphicsView):
    # ประกาศ signal เป็นตัวแปรคลาส
    history_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # เก็บรายการเส้น แยกตามประเภท
        self.h_lines = []  # เส้นแนวนอน
        self.v_lines = []  # เส้นแนวตั้ง
        
        self.current_line = None
        self.dragging_line = None
        self.drag_start_pos = QPointF(0, 0)
        
        # กำหนดสีและความหนาของเส้น
        self.h_line_thickness = 4  # ความหนาเส้นแนวนอนปกติ
        self.v_line_thickness = 4  # ความหนาเส้นแนวตั้งปกติ
        self.h_hover_thickness = 9  # ความหนาเส้นแนวนอนเมื่อ hover
        self.v_hover_thickness = 9  # ความหนาเส้นแนวตั้งเมื่อ hover
        self.user_color = QColor(0, 255, 0, 200)  # เส้นผู้ใช้สีเขียว
        self.v_line_color = QColor(0, 127, 255, 200)  # เส้นแนวตั้งสีฟ้า
        self.hover_color = QColor(255, 165, 0)    # สีส้มเมื่อ hover
        
        # กำหนดสีพื้นหลัง (เริ่มต้นเป็นสีเทาอ่อน)
        self.bg_color = QColor(240, 240, 240)
        
        # ตั้งค่าการแสดงผล
        self.setRenderHint(QPainter.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        # เปิดใช้งานการเลื่อน
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        
        # ตั้งค่าการซูม
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(1, 1)
        
        # เพิ่มตัวแปรใหม่
        self.hovered_line = None
        self.hovered_line_type = None  # เก็บประเภทของเส้นที่กำลัง hover (h หรือ v)
        
        # เพิ่มประวัติการกระทำสำหรับ undo/redo
        self.action_history = []
        self.current_action_index = -1  # ดัชนีปัจจุบันในประวัติการกระทำ
        
    def set_background_color(self, color):
        """กำหนดสีพื้นหลังของ canvas"""
        self.bg_color = color
        self.scene.setBackgroundBrush(QBrush(self.bg_color))
        # อัพเดท view
        self.viewport().update()

    def set_image(self, pixmap):
        """ตั้งค่าภาพสำหรับแสดงผล"""
        self.scene.clear()
        self.h_lines = []
        self.v_lines = []
        self.scene.addPixmap(pixmap)
        rect = pixmap.rect()
        self.scene.setSceneRect(QRectF(rect))
        # ตั้งค่าสีพื้นหลัง
        self.scene.setBackgroundBrush(QBrush(self.bg_color))
        # ปรับขนาดให้พอดีกับความกว้างของ view
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
        # รีเซ็ตประวัติการกระทำเมื่อโหลดภาพใหม่
        self.action_history = []
        self.current_action_index = -1

    def wheelEvent(self, event):
        """จัดการการเลื่อนลูกกลิ้ง"""
        if event.modifiers() == Qt.ControlModifier:
            # ซูมเมื่อกด Ctrl + ลูกกลิ้ง
            factor = 1.1 if event.angleDelta().y() > 0 else 0.9
            self.scale(factor, factor)
        else:
            # เลื่อนภาพขึ้น/ลง
            delta = event.angleDelta().y()
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta
            )

    def resizeEvent(self, event):
        """จัดการเมื่อขนาดหน้าต่างเปลี่ยน"""
        super().resizeEvent(event)
        if not self.scene.items():
            return
        # ปรับขนาดให้พอดีกับความกว้างของ view
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        """จัดการการคลิกเมาส์"""
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            
            # ตรวจสอบ Ctrl+Shift+คลิกซ้าย
            if event.modifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
                # Ctrl+Shift+คลิกซ้าย เพื่อเพิ่มเส้นแนวตั้ง
                self.add_vertical_line(pos.x())
            # ตรวจสอบ Shift+คลิกซ้าย (แต่ไม่มี Ctrl)
            elif event.modifiers() == Qt.ShiftModifier:
                # Shift+คลิกซ้าย เพื่อเพิ่มเส้นแนวนอน
                self.add_horizontal_line(pos.y())
            else:
                # เริ่มลากเส้นหรือเลื่อนภาพ
                # ตรวจหาเส้นแนวนอน
                h_line = self.find_nearby_horizontal_line(pos.y())
                # ตรวจหาเส้นแนวตั้ง
                v_line = self.find_nearby_vertical_line(pos.x())
                
                # ถ้าเจอทั้งสองประเภท เลือกเส้นที่ใกล้กว่า
                if h_line and v_line:
                    h_dist = abs(h_line["y"] - pos.y())
                    v_dist = abs(v_line["x"] - pos.x())
                    if h_dist <= v_dist:
                        self.dragging_line = h_line
                        self.hovered_line_type = "h"
                    else:
                        self.dragging_line = v_line
                        self.hovered_line_type = "v"
                elif h_line:
                    self.dragging_line = h_line
                    self.hovered_line_type = "h"
                elif v_line:
                    self.dragging_line = v_line
                    self.hovered_line_type = "v"
                
                if self.dragging_line:
                    self.drag_start_pos = pos
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """จัดการการเคลื่อนที่ของเมาส์"""
        if self.dragging_line:
            # โค้ดสำหรับการลากเส้น
            pos = self.mapToScene(event.position().toPoint())
            
            if self.hovered_line_type == "h":
                # ลากเส้นแนวนอน
                dy = pos.y() - self.drag_start_pos.y()
                line = self.dragging_line["line"]
                new_y = self.dragging_line["y"] + dy
                # ใช้ความยาวเส้นเดิมที่ล้นออกไปนอกขอบภาพ
                line.setLine(-500, new_y, self.scene.width() + 500, new_y)
            else:
                # ลากเส้นแนวตั้ง
                dx = pos.x() - self.drag_start_pos.x()
                line = self.dragging_line["line"]
                new_x = self.dragging_line["x"] + dx
                # ใช้ความยาวเส้นเดิมที่ล้นออกไปนอกขอบภาพ
                line.setLine(new_x, -500, new_x, self.scene.height() + 500)
        else:
            # เช็คว่าเมาส์อยู่เหนือเส้นใดๆ หรือไม่
            pos = self.mapToScene(event.position().toPoint())
            
            # ตรวจหาเส้นแนวนอนที่ใกล้เมาส์
            h_line = self.find_nearby_horizontal_line(pos.y())
            # ตรวจหาเส้นแนวตั้งที่ใกล้เมาส์
            v_line = self.find_nearby_vertical_line(pos.x())
            
            # ตรวจสอบว่ามีเส้นที่ใกล้กว่ากัน
            if h_line and v_line:
                h_dist = abs(h_line["y"] - pos.y())
                v_dist = abs(v_line["x"] - pos.x())
                if h_dist <= v_dist:
                    current_line = h_line
                    line_type = "h"
                else:
                    current_line = v_line
                    line_type = "v"
            elif h_line:
                current_line = h_line
                line_type = "h"
            elif v_line:
                current_line = v_line
                line_type = "v"
            else:
                current_line = None
                line_type = None
            
            # ถ้าเมาส์อยู่เหนือเส้นที่ต่างจากเดิม
            if current_line != self.hovered_line or line_type != self.hovered_line_type:
                # คืนค่าเส้นเก่ากลับเป็นปกติ
                if self.hovered_line:
                    old_line = self.hovered_line["line"]
                    if self.hovered_line_type == "h":
                        old_line.setPen(QPen(self.user_color, self.h_line_thickness))
                    else:
                        old_line.setPen(QPen(self.v_line_color, self.v_line_thickness))
                
                # ไฮไลท์เส้นใหม่
                if current_line:
                    current_line["line"].setPen(QPen(self.hover_color, self.h_hover_thickness))
                
                self.hovered_line = current_line
                self.hovered_line_type = line_type
            
            super().mouseMoveEvent(event)
        
        # เปลี่ยน cursor ตามทิศทางของเส้น
        if self.hovered_line:
            if self.hovered_line_type == "h":
                self.setCursor(Qt.SizeVerCursor)
            else:
                self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        """จัดการการปล่อยเมาส์"""
        if self.dragging_line:
            pos = self.mapToScene(event.position().toPoint())
            
            if self.hovered_line_type == "h":
                # เส้นแนวนอน
                old_y = self.dragging_line["y"]
                new_y = old_y + (pos.y() - self.drag_start_pos.y())
                
                # บันทึกการกระทำสำหรับ undo (เฉพาะเมื่อมีการเปลี่ยนแปลงจริง)
                if abs(old_y - new_y) > 0.001:  # ถ้ามีการเปลี่ยนแปลงจริง
                    line_index = self.h_lines.index(self.dragging_line)
                    self.add_action(LineAction(LineAction.MOVE, line_index, None, old_y, new_y))
                
                self.dragging_line["y"] = new_y
                
                # คืนค่าสีเส้นเป็นปกติ
                line = self.dragging_line["line"]
                line.setPen(QPen(self.user_color, self.h_line_thickness))
            else:
                # เส้นแนวตั้ง
                old_x = self.dragging_line["x"]
                new_x = old_x + (pos.x() - self.drag_start_pos.x())
                
                # บันทึกการกระทำสำหรับ undo (เฉพาะเมื่อมีการเปลี่ยนแปลงจริง)
                if abs(old_x - new_x) > 0.001:  # ถ้ามีการเปลี่ยนแปลงจริง
                    line_index = self.v_lines.index(self.dragging_line)
                    self.add_action(LineAction(LineAction.MOVE, line_index, None, old_x, new_x))
                
                self.dragging_line["x"] = new_x
                
                # คืนค่าสีเส้นเป็นปกติ
                line = self.dragging_line["line"]
                line.setPen(QPen(self.v_line_color, self.v_line_thickness))
            
            self.dragging_line = None
            self.hovered_line_type = None
        else:
            super().mouseReleaseEvent(event)

    def add_horizontal_line(self, y):
        """เพิ่มเส้นแนวนอนโดยผู้ใช้"""
        width = self.scene.width()
        # ให้เส้นล้นออกไปนอกขอบภาพทั้งซ้ายและขวา 500 พิกเซลเพื่อให้เห็นชัดเจน
        line = QGraphicsLineItem(-500, y, width + 500, y)
        line.setPen(QPen(self.user_color, self.h_line_thickness))
        self.scene.addItem(line)
        line_data = {"line": line, "type": LineAction.HORIZONTAL, "y": y}
        self.h_lines.append(line_data)
        
        # บันทึกการกระทำสำหรับ undo
        self.add_action(LineAction(LineAction.ADD, len(self.h_lines) - 1, line_data))
        
        return line_data

    def add_vertical_line(self, x):
        """เพิ่มเส้นแนวตั้งโดยผู้ใช้"""
        height = self.scene.height()
        # ให้เส้นล้นออกไปนอกขอบภาพทั้งบนและล่าง 500 พิกเซลเพื่อให้เห็นชัดเจน
        line = QGraphicsLineItem(x, -500, x, height + 500)
        line.setPen(QPen(self.v_line_color, self.v_line_thickness))
        self.scene.addItem(line)
        line_data = {"line": line, "type": LineAction.VERTICAL, "x": x}
        self.v_lines.append(line_data)
        
        # บันทึกการกระทำสำหรับ undo
        self.add_action(LineAction(LineAction.ADD, len(self.v_lines) - 1, line_data))
        
        return line_data

    def find_nearby_horizontal_line(self, y, threshold=10):
        """หาเส้นแนวนอนที่อยู่ใกล้จุดที่คลิก"""
        view_threshold = threshold / self.transform().m22()  # ปรับขนาดตามการซูม
        for line_info in self.h_lines:
            if abs(line_info["y"] - y) < view_threshold:
                return line_info
        return None

    def find_nearby_vertical_line(self, x, threshold=10):
        """หาเส้นแนวตั้งที่อยู่ใกล้จุดที่คลิก"""
        view_threshold = threshold / self.transform().m11()  # ปรับขนาดตามการซูม
        for line_info in self.v_lines:
            if abs(line_info["x"] - x) < view_threshold:
                return line_info
        return None

    def get_horizontal_breakpoints(self):
        """รับค่าจุดตัดแนวนอนที่ผู้ใช้กำหนด"""
        return sorted([line["y"] for line in self.h_lines])
        
    def get_vertical_breakpoints(self):
        """รับค่าจุดตัดแนวตั้งที่ผู้ใช้กำหนด"""
        return sorted([line["x"] for line in self.v_lines])

    def keyPressEvent(self, event):
        """จัดการการกดปุ่ม"""
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            # ลบเส้นที่เลือก
            if self.hovered_line:
                if self.hovered_line_type == "h":
                    line_index = self.h_lines.index(self.hovered_line)
                    line_data = self.hovered_line
                    
                    # บันทึกการกระทำสำหรับ undo
                    self.add_action(LineAction(LineAction.REMOVE, line_index, line_data))
                    
                    self.scene.removeItem(self.hovered_line["line"])
                    self.h_lines.remove(self.hovered_line)
                else:
                    line_index = self.v_lines.index(self.hovered_line)
                    line_data = self.hovered_line
                    
                    # บันทึกการกระทำสำหรับ undo
                    self.add_action(LineAction(LineAction.REMOVE, line_index, line_data))
                    
                    self.scene.removeItem(self.hovered_line["line"])
                    self.v_lines.remove(self.hovered_line)
                
                self.hovered_line = None
                self.hovered_line_type = None
                return
        # เพิ่มการจัดการปุ่มลัด Ctrl+Z (Undo) และ Ctrl+Y (Redo)
        elif event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_Z:
                self.undo()
                return
            elif event.key() == Qt.Key_Y:
                self.redo()
                return
        
        super().keyPressEvent(event)

    def add_action(self, action):
        """เพิ่มการกระทำลงในประวัติสำหรับ undo/redo"""
        # ลบประวัติหลังจากตำแหน่งปัจจุบัน (ในกรณีที่มีการ undo แล้วทำการกระทำใหม่)
        if self.current_action_index < len(self.action_history) - 1:
            self.action_history = self.action_history[:self.current_action_index + 1]
        
        # เพิ่มการกระทำใหม่
        self.action_history.append(action)
        self.current_action_index = len(self.action_history) - 1
        
        # แจ้งเตือนว่ามีการเปลี่ยนแปลงในประวัติ
        self.history_changed.emit()

    def undo(self):
        """ย้อนการกระทำล่าสุด"""
        if self.current_action_index >= 0:
            action = self.action_history[self.current_action_index]
            
            if action.type == LineAction.ADD:
                # ย้อนการเพิ่มเส้น
                if action.line_data["type"] == LineAction.HORIZONTAL:
                    line_data = self.h_lines[action.line_index]
                    self.scene.removeItem(line_data["line"])
                    self.h_lines.pop(action.line_index)
                    
                    # อัพเดทเส้นที่กำลัง hover (ถ้ามี)
                    if self.hovered_line == line_data:
                        self.hovered_line = None
                        self.hovered_line_type = None
                else:
                    line_data = self.v_lines[action.line_index]
                    self.scene.removeItem(line_data["line"])
                    self.v_lines.pop(action.line_index)
                    
                    # อัพเดทเส้นที่กำลัง hover (ถ้ามี)
                    if self.hovered_line == line_data:
                        self.hovered_line = None
                        self.hovered_line_type = None
            
            elif action.type == LineAction.REMOVE:
                # ย้อนการลบเส้น
                line_data = action.line_data
                
                if line_data["type"] == LineAction.HORIZONTAL:
                    width = self.scene.width()
                    
                    # สร้างเส้นใหม่ทดแทนเส้นที่ลบไป
                    new_line = QGraphicsLineItem(-500, line_data["y"], width + 500, line_data["y"])
                    new_line.setPen(QPen(self.user_color, self.h_line_thickness))
                    self.scene.addItem(new_line)
                    
                    # ปรับปรุงข้อมูลเส้น
                    line_data["line"] = new_line
                    
                    # แทรกเส้นกลับเข้าไปที่ตำแหน่งเดิม
                    if action.line_index <= len(self.h_lines):
                        self.h_lines.insert(action.line_index, line_data)
                    else:
                        self.h_lines.append(line_data)
                else:
                    height = self.scene.height()
                    
                    # สร้างเส้นใหม่ทดแทนเส้นที่ลบไป
                    new_line = QGraphicsLineItem(line_data["x"], -500, line_data["x"], height + 500)
                    new_line.setPen(QPen(self.v_line_color, self.v_line_thickness))
                    self.scene.addItem(new_line)
                    
                    # ปรับปรุงข้อมูลเส้น
                    line_data["line"] = new_line
                    
                    # แทรกเส้นกลับเข้าไปที่ตำแหน่งเดิม
                    if action.line_index <= len(self.v_lines):
                        self.v_lines.insert(action.line_index, line_data)
                    else:
                        self.v_lines.append(line_data)
            
            elif action.type == LineAction.MOVE:
                # ย้อนการย้ายเส้น
                line_info = action.line_data
                
                if line_info["type"] == LineAction.HORIZONTAL:
                    line_data = self.h_lines[action.line_index]
                    line = line_data["line"]
                    
                    # นำเส้นกลับไปยังตำแหน่งเดิม
                    line.setLine(-500, action.old_pos, self.scene.width() + 500, action.old_pos)
                    line_data["y"] = action.old_pos
                else:
                    line_data = self.v_lines[action.line_index]
                    line = line_data["line"]
                    
                    # นำเส้นกลับไปยังตำแหน่งเดิม
                    line.setLine(action.old_pos, -500, action.old_pos, self.scene.height() + 500)
                    line_data["x"] = action.old_pos
            
            # เลื่อนดัชนีปัจจุบันลง
            self.current_action_index -= 1
            
            # แจ้งเตือนว่ามีการเปลี่ยนแปลงในประวัติ
            self.history_changed.emit()

    def redo(self):
        """ทำซ้ำการกระทำที่ย้อน"""
        if self.current_action_index < len(self.action_history) - 1:
            # เลื่อนดัชนีขึ้น
            self.current_action_index += 1
            action = self.action_history[self.current_action_index]
            
            if action.type == LineAction.ADD:
                # ทำซ้ำการเพิ่มเส้น
                line_data = action.line_data
                
                if line_data["type"] == LineAction.HORIZONTAL:
                    width = self.scene.width()
                    y = line_data["y"]
                    
                    # สร้างเส้นใหม่
                    new_line = QGraphicsLineItem(-500, y, width + 500, y)
                    new_line.setPen(QPen(self.user_color, self.h_line_thickness))
                    self.scene.addItem(new_line)
                    
                    # สร้างข้อมูลเส้น
                    new_line_data = {"line": new_line, "type": LineAction.HORIZONTAL, "y": y}
                    
                    # เพิ่มเส้นลงในรายการ
                    if action.line_index <= len(self.h_lines):
                        self.h_lines.insert(action.line_index, new_line_data)
                    else:
                        self.h_lines.append(new_line_data)
                else:
                    height = self.scene.height()
                    x = line_data["x"]
                    
                    # สร้างเส้นใหม่
                    new_line = QGraphicsLineItem(x, -500, x, height + 500)
                    new_line.setPen(QPen(self.v_line_color, self.v_line_thickness))
                    self.scene.addItem(new_line)
                    
                    # สร้างข้อมูลเส้น
                    new_line_data = {"line": new_line, "type": LineAction.VERTICAL, "x": x}
                    
                    # เพิ่มเส้นลงในรายการ
                    if action.line_index <= len(self.v_lines):
                        self.v_lines.insert(action.line_index, new_line_data)
                    else:
                        self.v_lines.append(new_line_data)
            
            elif action.type == LineAction.REMOVE:
                # ทำซ้ำการลบเส้น
                line_data = action.line_data
                
                if line_data["type"] == LineAction.HORIZONTAL:
                    if action.line_index < len(self.h_lines):
                        line_data = self.h_lines[action.line_index]
                        self.scene.removeItem(line_data["line"])
                        self.h_lines.pop(action.line_index)
                        
                        # อัพเดทเส้นที่กำลัง hover (ถ้ามี)
                        if self.hovered_line == line_data:
                            self.hovered_line = None
                            self.hovered_line_type = None
                else:
                    if action.line_index < len(self.v_lines):
                        line_data = self.v_lines[action.line_index]
                        self.scene.removeItem(line_data["line"])
                        self.v_lines.pop(action.line_index)
                        
                        # อัพเดทเส้นที่กำลัง hover (ถ้ามี)
                        if self.hovered_line == line_data:
                            self.hovered_line = None
                            self.hovered_line_type = None
            
            elif action.type == LineAction.MOVE:
                # ทำซ้ำการย้ายเส้น
                line_info = action.line_data
                
                if line_info["type"] == LineAction.HORIZONTAL:
                    if action.line_index < len(self.h_lines):
                        line_data = self.h_lines[action.line_index]
                        line = line_data["line"]
                        
                        # ย้ายเส้นไปยังตำแหน่งใหม่
                        line.setLine(-500, action.new_pos, self.scene.width() + 500, action.new_pos)
                        line_data["y"] = action.new_pos
                else:
                    if action.line_index < len(self.v_lines):
                        line_data = self.v_lines[action.line_index]
                        line = line_data["line"]
                        
                        # ย้ายเส้นไปยังตำแหน่งใหม่
                        line.setLine(action.new_pos, -500, action.new_pos, self.scene.height() + 500)
                        line_data["x"] = action.new_pos
            
            # แจ้งเตือนว่ามีการเปลี่ยนแปลงในประวัติ
            self.history_changed.emit()
            
    def can_undo(self):
        """ตรวจสอบว่าสามารถย้อนการกระทำได้หรือไม่"""
        return self.current_action_index >= 0
        
    def can_redo(self):
        """ตรวจสอบว่าสามารถทำซ้ำได้หรือไม่"""
        return self.current_action_index < len(self.action_history) - 1

    def set_line_properties(self, h_color, v_color, h_thickness, v_thickness):
        """กำหนดสีและความหนาของเส้น"""
        self.user_color = h_color
        self.v_line_color = v_color
        self.h_line_thickness = h_thickness
        self.v_line_thickness = v_thickness
        self.h_hover_thickness = h_thickness + 5  # เพิ่มความหนาอีก 5 px เมื่อ hover
        self.v_hover_thickness = v_thickness + 5  # เพิ่มความหนาอีก 5 px เมื่อ hover
        
        # อัพเดทเส้นที่มีอยู่แล้ว
        for line_info in self.h_lines:
            line = line_info["line"]
            if line_info == self.hovered_line and self.hovered_line_type == "h":
                line.setPen(QPen(self.hover_color, self.h_hover_thickness))
            else:
                line.setPen(QPen(self.user_color, self.h_line_thickness))
        
        for line_info in self.v_lines:
            line = line_info["line"]
            if line_info == self.hovered_line and self.hovered_line_type == "v":
                line.setPen(QPen(self.hover_color, self.v_hover_thickness))
            else:
                line.setPen(QPen(self.v_line_color, self.v_line_thickness))
                
        # อัพเดทจอแสดงผล
        self.scene.update()

class ManualSplitDialog(QDialog):
    def __init__(self, parent=None, pixmap=None):
        super().__init__(parent)
        self.setWindowTitle("ตัวตัดภาพด้วยตนเอง")
        self.setMinimumSize(800, 600)
        
        # กำหนดค่าเริ่มต้น
        self.pixmap = pixmap
        self.h_line_color = QColor(255, 0, 0, 150)  # สีแดงกึ่งโปร่งใส
        self.v_line_color = QColor(0, 0, 255, 150)  # สีน้ำเงินกึ่งโปร่งใส
        self.line_thickness = 4  # ความหนาของเส้นทั่วไป
        self.h_line_thickness = 15  # ความหนาของเส้นแนวนอน
        self.v_line_thickness = 8  # ความหนาของเส้นแนวตั้ง
        self.bg_color = QColor(240, 240, 240)  # สีพื้นหลังเทาอ่อน
        
        self.resize_factor = 100
        self.scale_factor = 1.0
        
        self.setup_ui()
        
        # ตั้งค่าเริ่มต้นสำหรับ view
        if self.pixmap:
            self.view.set_image(self.pixmap)
            # กำหนดสีและความหนาของเส้นให้กับ view
            self.view.set_line_properties(self.h_line_color, self.v_line_color, self.h_line_thickness, self.v_line_thickness)
            # กำหนดสีพื้นหลังให้กับ view
            self.view.set_background_color(self.bg_color)
            
        # เชื่อมต่อสัญญาณการเปลี่ยนแปลงประวัติ
        self.view.history_changed.connect(self.update_history_buttons)
        
        # อัพเดทสถานะของปุ่มครั้งแรก
        self.update_history_buttons()

    def setup_ui(self):
        # สร้าง layout หลัก
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)  # ลบขอบเพื่อใช้พื้นที่ให้เต็มที่
        main_layout.setSpacing(0)  # ลดช่องว่างระหว่างองค์ประกอบ
        
        # สร้าง splitter เพื่อแบ่งพื้นที่ระหว่าง canvas และควบคุม
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(3)  # handle บางลงเพื่อดูเรียบร้อย
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #dddddd;
            }
            QSplitter::handle:hover {
                background-color: #2196F3;
            }
        """)
        
        # --- ส่วนแสดงผลภาพ (ตั้งให้อยู่ด้านบน) ---
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        self.view = ManualSplitView()
        self.view.setFrameShape(QFrame.NoFrame)  # ลบกรอบเพื่อดูเรียบร้อย
        self.view.setMinimumHeight(500)
        canvas_layout.addWidget(self.view)
        
        # --- ส่วนควบคุม (ตั้งให้อยู่ด้านล่าง) ---
        controls_container = QWidget()
        controls_container.setMinimumHeight(200)  # กำหนดความสูงขั้นต่ำของส่วนควบคุม
        controls_container.setMaximumHeight(250)  # จำกัดความสูงของส่วนควบคุม
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(10, 5, 10, 5)
        controls_layout.setSpacing(5)
        
        # แถบควบคุมด้านบน (คำแนะนำและ Undo/Redo)
        top_bar = QWidget()
        top_bar.setStyleSheet("background-color: #f0f0f0; border-radius: 5px;")
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(8, 2, 8, 2)
        
        # คำแนะนำการใช้งาน
        instruction_label = QLabel(
            "คำแนะนำ: Shift+คลิกซ้าย = เส้นแนวนอน | Ctrl+Shift+คลิกซ้าย = เส้นแนวตั้ง | Delete = ลบเส้น | Ctrl+Z = ยกเลิก | Ctrl+Y = ทำซ้ำ | Ctrl+เลื่อนเมาส์ = ซูม"
        )
        instruction_label.setStyleSheet("color: #444; font-size: 11px;")
        top_bar_layout.addWidget(instruction_label)
        
        # ปุ่ม Undo/Redo
        self.undo_btn = QPushButton("↩")
        self.undo_btn.setToolTip("ย้อนกลับ (Undo)")
        self.undo_btn.setFixedSize(28, 28)
        self.undo_btn.clicked.connect(self.undo_action)
        self.undo_btn.setStyleSheet("""
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
        """)
        top_bar_layout.addWidget(self.undo_btn)
        
        self.redo_btn = QPushButton("↪")
        self.redo_btn.setToolTip("ทำซ้ำ (Redo)")
        self.redo_btn.setFixedSize(28, 28)
        self.redo_btn.clicked.connect(self.redo_action)
        self.redo_btn.setStyleSheet("""
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
        """)
        top_bar_layout.addWidget(self.redo_btn)
        
        controls_layout.addWidget(top_bar)
        
        # แท็บควบคุม - ใช้ QTabWidget ที่มี style แบบ modern
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)
        tabs.setStyleSheet("""
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
        """)
        
        # --- แท็บ 1: ตั้งค่าเส้น ---
        line_tab = QWidget()
        line_layout = QHBoxLayout(line_tab)
        line_layout.setContentsMargins(8, 5, 8, 5)
        line_layout.setSpacing(8)
        
        # ตั้งค่าสีเส้นแนวนอน
        h_color_label = QLabel("สีเส้นแนวนอน:")
        h_color_label.setStyleSheet("color: #444444;")
        line_layout.addWidget(h_color_label)
        
        self.h_color_btn = QPushButton()
        self.h_color_btn.setFixedSize(30, 25)
        self.h_color_btn.setCursor(Qt.PointingHandCursor)
        self.update_h_color_button()
        self.h_color_btn.clicked.connect(self.select_h_color)
        line_layout.addWidget(self.h_color_btn)
        
        # ตั้งค่าสีเส้นแนวตั้ง
        v_color_label = QLabel("สีเส้นแนวตั้ง:")
        v_color_label.setStyleSheet("color: #444444;")
        line_layout.addWidget(v_color_label)
        
        self.v_color_btn = QPushButton()
        self.v_color_btn.setFixedSize(30, 25)
        self.v_color_btn.setCursor(Qt.PointingHandCursor)
        self.update_v_color_button()
        self.v_color_btn.clicked.connect(self.select_v_color)
        line_layout.addWidget(self.v_color_btn)
        
        # ตั้งค่าความหนาเส้นแนวนอน
        h_thickness_label = QLabel("ความหนาเส้นแนวนอน:")
        h_thickness_label.setStyleSheet("color: #444444;")
        line_layout.addWidget(h_thickness_label)
        
        self.h_thickness_slider = QSlider(Qt.Horizontal)
        self.h_thickness_slider.setMinimum(2)
        self.h_thickness_slider.setMaximum(30)
        self.h_thickness_slider.setValue(self.h_line_thickness)
        self.h_thickness_slider.setTickPosition(QSlider.TicksBelow)
        self.h_thickness_slider.setTickInterval(2)
        self.h_thickness_slider.setStyleSheet("""
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
        """)
        self.h_thickness_slider.valueChanged.connect(self.change_h_thickness)
        line_layout.addWidget(self.h_thickness_slider)
        
        self.h_thickness_value = QLabel(f"{self.h_line_thickness}px")
        self.h_thickness_value.setStyleSheet("color: #444444; min-width: 35px;")
        line_layout.addWidget(self.h_thickness_value)
        
        # ตั้งค่าความหนาเส้นแนวตั้ง
        v_thickness_label = QLabel("ความหนาเส้นแนวตั้ง:")
        v_thickness_label.setStyleSheet("color: #444444;")
        line_layout.addWidget(v_thickness_label)
        
        self.v_thickness_slider = QSlider(Qt.Horizontal)
        self.v_thickness_slider.setMinimum(2)
        self.v_thickness_slider.setMaximum(30)
        self.v_thickness_slider.setValue(self.v_line_thickness)
        self.v_thickness_slider.setTickPosition(QSlider.TicksBelow)
        self.v_thickness_slider.setTickInterval(2)
        self.v_thickness_slider.setStyleSheet("""
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
        """)
        self.v_thickness_slider.valueChanged.connect(self.change_v_thickness)
        line_layout.addWidget(self.v_thickness_slider)
        
        self.v_thickness_value = QLabel(f"{self.v_line_thickness}px")
        self.v_thickness_value.setStyleSheet("color: #444444; min-width: 35px;")
        line_layout.addWidget(self.v_thickness_value)
        
        # Apply ปุ่ม
        self.apply_line_settings_btn = QPushButton("ปรับใช้")
        self.apply_line_settings_btn.setCursor(Qt.PointingHandCursor)
        self.apply_line_settings_btn.clicked.connect(self.apply_line_settings)
        self.apply_line_settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 3px 10px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3d8b40; }
        """)
        line_layout.addWidget(self.apply_line_settings_btn)
        
        tabs.addTab(line_tab, "ตั้งค่าเส้น")
        
        # --- แท็บ 2: พื้นหลัง ---
        bg_tab = QWidget()
        bg_layout = QHBoxLayout(bg_tab)
        bg_layout.setContentsMargins(8, 5, 8, 5)
        bg_layout.setSpacing(8)
        
        bg_color_label = QLabel("สีพื้นหลัง:")
        bg_color_label.setStyleSheet("color: #444444;")
        bg_layout.addWidget(bg_color_label)
        
        self.bg_color_btn = QPushButton()
        self.bg_color_btn.setFixedSize(30, 25)
        self.bg_color_btn.setCursor(Qt.PointingHandCursor)
        self.update_bg_color_button()
        self.bg_color_btn.clicked.connect(self.select_bg_color)
        bg_layout.addWidget(self.bg_color_btn)
        
        # ปุ่มสีพื้นหลังสำเร็จรูป
        color_btn_style = """
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
        
        self.default_bg_btn = QPushButton("สีเริ่มต้น")
        self.default_bg_btn.setCursor(Qt.PointingHandCursor)
        self.default_bg_btn.clicked.connect(self.set_default_bg_color)
        self.default_bg_btn.setStyleSheet(color_btn_style)
        bg_layout.addWidget(self.default_bg_btn)
        
        self.white_bg_btn = QPushButton("สีขาว")
        self.white_bg_btn.setCursor(Qt.PointingHandCursor)
        self.white_bg_btn.clicked.connect(self.set_white_bg_color)
        self.white_bg_btn.setStyleSheet(color_btn_style)
        bg_layout.addWidget(self.white_bg_btn)
        
        self.black_bg_btn = QPushButton("สีดำ")
        self.black_bg_btn.setCursor(Qt.PointingHandCursor)
        self.black_bg_btn.clicked.connect(self.set_black_bg_color)
        self.black_bg_btn.setStyleSheet(color_btn_style)
        bg_layout.addWidget(self.black_bg_btn)
        
        bg_layout.addStretch()
        
        tabs.addTab(bg_tab, "พื้นหลัง")
        
        # --- แท็บ 3: ขนาดภาพ ---
        resize_tab = QWidget()
        resize_layout = QHBoxLayout(resize_tab)
        resize_layout.setContentsMargins(8, 5, 8, 5)
        resize_layout.setSpacing(8)
        
        # เปลี่ยนเป็นใช้ Slider แทน spinbox เพื่อให้ใช้งานง่ายและสวยงาม
        resize_label = QLabel("ขนาดภาพ:")
        resize_label.setStyleSheet("color: #444444;")
        resize_layout.addWidget(resize_label)
        
        self.resize_slider = QSlider(Qt.Horizontal)
        self.resize_slider.setMinimum(10)
        self.resize_slider.setMaximum(200)  # เพิ่มช่วงให้ซูมได้มากขึ้น
        self.resize_slider.setValue(100)
        self.resize_slider.setTickPosition(QSlider.TicksBelow)
        self.resize_slider.setTickInterval(10)
        self.resize_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #dddddd;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #1976D2;
            }
        """)
        resize_layout.addWidget(self.resize_slider)
        
        self.resize_value = QLabel("100%")
        self.resize_value.setFixedWidth(50)
        self.resize_value.setStyleSheet("color: #444444;")
        resize_layout.addWidget(self.resize_value)
        
        # เชื่อมต่อสัญญาณเพื่ออัพเดตข้อความ
        self.resize_slider.valueChanged.connect(self.update_resize_value)
        
        self.apply_resize_btn = QPushButton("ปรับขนาด")
        self.apply_resize_btn.setCursor(Qt.PointingHandCursor)
        self.apply_resize_btn.clicked.connect(self.resize_image)
        self.apply_resize_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 3px 10px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:pressed { background-color: #0D47A1; }
        """)
        resize_layout.addWidget(self.apply_resize_btn)
        
        # ปุ่มพอดีกับหน้าจอ (fit to screen)
        self.fit_screen_btn = QPushButton("พอดีจอ")
        self.fit_screen_btn.setCursor(Qt.PointingHandCursor)
        self.fit_screen_btn.clicked.connect(self.fit_to_screen)
        self.fit_screen_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                padding: 3px 10px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #F57C00; }
            QPushButton:pressed { background-color: #E65100; }
        """)
        resize_layout.addWidget(self.fit_screen_btn)
        
        resize_layout.addStretch()
        
        tabs.addTab(resize_tab, "ขนาดภาพ")
        
        controls_layout.addWidget(tabs)
        
        # แถบปุ่มด้านล่าง
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(10, 5, 10, 5)
        button_layout.addStretch()
        
        cancel_btn = QPushButton("ยกเลิก")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                padding: 8px 16px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #d0d0d0; }
            QPushButton:pressed { background-color: #c0c0c0; }
        """)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("ตกลง")
        ok_btn.setCursor(Qt.PointingHandCursor)
        ok_btn.clicked.connect(self.accept)
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:pressed { background-color: #0D47A1; }
        """)
        button_layout.addWidget(ok_btn)
        
        controls_layout.addLayout(button_layout)
        
        # เพิ่ม widget ไปยัง splitter
        splitter.addWidget(canvas_container)
        splitter.addWidget(controls_container)
        
        # กำหนดอัตราส่วนระหว่าง canvas และควบคุม
        splitter.setSizes([800, 220])
        
        # เพิ่ม splitter ไปยัง layout หลัก
        main_layout.addWidget(splitter)
        
        # ตั้งค่าขนาดหน้าต่าง - ใช้เต็มหน้าจอเกือบทั้งหมด
        screen = QApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(800, 600)
        width = int(screen.width() * 0.9)
        height = int(screen.height() * 0.9)
        self.resize(width, height)
        
        # กำหนด layout หลักให้กับ dialog
        self.setLayout(main_layout)

    def update_resize_value(self, value):
        """อัพเดตข้อความแสดงค่าการปรับขนาด"""
        self.resize_value.setText(f"{value}%")

    def fit_to_screen(self):
        """ปรับขนาดภาพให้พอดีกับหน้าจอ"""
        if self.pixmap:
            # ทำการรีเซ็ตขนาดภาพเป็น 100% ก่อน
            self.resize_slider.setValue(100)
            
            # โหลดภาพที่ขนาด 100%
            self.view.set_image(self.pixmap)
            
            # ปรับขนาดให้พอดีกับหน้าจอ
            self.view.fitInView(self.view.scene.sceneRect(), Qt.KeepAspectRatio)
            
            # อัพเดตค่า scale factor ให้สะท้อนการปรับขนาด
            transform = self.view.transform()
            self.scale_factor = min(transform.m11(), transform.m22())
            
            # อัพเดตข้อความแสดงขนาด
            self.resize_slider.blockSignals(True)
            self.resize_slider.setValue(100)
            self.resize_slider.blockSignals(False)
            self.resize_value.setText("100%")

    def resize_image(self):
        """ย่อขนาดภาพตามที่ผู้ใช้กำหนด"""
        if self.pixmap:
            scale = self.resize_slider.value() / 100.0
            if scale == 1.0:
                # ใช้ภาพต้นฉบับโดยไม่มีการย่อขนาด
                self.view.set_image(self.pixmap)
            else:
                # คำนวณขนาดใหม่
                new_width = int(self.pixmap.width() * scale)
                new_height = int(self.pixmap.height() * scale)
                
                # สร้างภาพใหม่ที่ถูกย่อขนาด
                scaled_pixmap = self.pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.view.set_image(scaled_pixmap)
            
            # อัพเดตค่า scale factor
            self.scale_factor = scale

    def get_scale_factor(self):
        """รับค่าสัดส่วนการย่อขนาด"""
        return self.resize_slider.value() / 100.0
        
    def get_horizontal_breakpoints(self):
        """รับค่าจุดตัดแนวนอนที่ผู้ใช้กำหนด"""
        return self.view.get_horizontal_breakpoints()
        
    def get_vertical_breakpoints(self):
        """รับค่าจุดตัดแนวตั้งที่ผู้ใช้กำหนด"""
        return self.view.get_vertical_breakpoints()

    def keyPressEvent(self, event):
        """จัดการการกดปุ่ม - ปิดการทำงานของ ESC"""
        if event.key() == Qt.Key_Escape:
            # ไม่ทำอะไรเมื่อกด ESC เพื่อป้องกันการปิด dialog โดยไม่ตั้งใจ
            event.ignore()
        else:
            super().keyPressEvent(event)

    def undo_action(self):
        """เรียกใช้คำสั่ง Undo ของ view"""
        self.view.undo()
        self.update_history_buttons()
        
    def redo_action(self):
        """เรียกใช้คำสั่ง Redo ของ view"""
        self.view.redo()
        self.update_history_buttons()
        
    def update_history_buttons(self):
        """อัพเดทสถานะของปุ่ม Undo/Redo"""
        self.undo_btn.setEnabled(self.view.can_undo())
        self.redo_btn.setEnabled(self.view.can_redo())

    def update_h_color_button(self):
        """อัพเดทสีของปุ่มเลือกสีเส้นแนวนอน"""
        style = f"""
            QPushButton {{
                background-color: {self.h_line_color.name()};
                border: 1px solid #bbb;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                border: 1px solid #888;
            }}
        """
        self.h_color_btn.setStyleSheet(style)

    def update_v_color_button(self):
        """อัพเดทสีของปุ่มเลือกสีเส้นแนวตั้ง"""
        style = f"""
            QPushButton {{
                background-color: {self.v_line_color.name()};
                border: 1px solid #bbb;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                border: 1px solid #888;
            }}
        """
        self.v_color_btn.setStyleSheet(style)

    def select_h_color(self):
        """เปิดหน้าต่างเลือกสีสำหรับเส้นแนวนอน"""
        color = QColorDialog.getColor(self.h_line_color, self, "เลือกสีเส้นตัดแนวนอน", QColorDialog.ShowAlphaChannel)
        if color.isValid():
            self.h_line_color = color
            self.update_h_color_button()

    def select_v_color(self):
        """เปิดหน้าต่างเลือกสีสำหรับเส้นแนวตั้ง"""
        color = QColorDialog.getColor(self.v_line_color, self, "เลือกสีเส้นตัดแนวตั้ง", QColorDialog.ShowAlphaChannel)
        if color.isValid():
            self.v_line_color = color
            self.update_v_color_button()

    def change_h_thickness(self, value):
        """อัพเดทข้อความแสดงความหนาเส้นแนวนอน"""
        self.h_line_thickness = value
        self.h_thickness_value.setText(f"{value}px")
        self.apply_line_settings()

    def change_v_thickness(self, value):
        """อัพเดทข้อความแสดงความหนาเส้นแนวตั้ง"""
        self.v_line_thickness = value
        self.v_thickness_value.setText(f"{value}px")
        self.apply_line_settings()

    def change_thickness(self, value):
        """อัพเดทข้อความแสดงความหนา (เมทอดเดิมให้คงไว้เพื่อความเข้ากันได้)"""
        self.line_thickness = value
        self.h_line_thickness = value
        self.v_line_thickness = value
        self.thickness_value.setText(f"{value}px")

    def apply_line_settings(self):
        """ปรับใช้การตั้งค่าเส้นกับ view"""
        self.view.set_line_properties(self.h_line_color, self.v_line_color, self.h_line_thickness, self.v_line_thickness)

    def update_bg_color_button(self):
        """อัพเดทสีของปุ่มเลือกสีพื้นหลัง"""
        style = f"""
            QPushButton {{
                background-color: {self.bg_color.name()};
                border: 1px solid #bbb;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                border: 1px solid #888;
            }}
        """
        self.bg_color_btn.setStyleSheet(style)

    def select_bg_color(self):
        """เปิดหน้าต่างเลือกสีพื้นหลัง"""
        color = QColorDialog.getColor(self.bg_color, self, "เลือกสีพื้นหลัง", QColorDialog.ShowAlphaChannel)
        if color.isValid():
            self.bg_color = color
            self.update_bg_color_button()
            # อัพเดตสีพื้นหลังทันที
            self.view.set_background_color(self.bg_color)

    def set_default_bg_color(self):
        """ตั้งค่าสีพื้นหลังเริ่มต้น"""
        self.bg_color = QColor(240, 240, 240)
        self.update_bg_color_button()
        # อัพเดตสีพื้นหลังทันที
        self.view.set_background_color(self.bg_color)

    def set_white_bg_color(self):
        """ตั้งค่าสีพื้นหลังเป็นสีขาว"""
        self.bg_color = QColor(255, 255, 255)
        self.update_bg_color_button()
        # อัพเดตสีพื้นหลังทันที
        self.view.set_background_color(self.bg_color)

    def set_black_bg_color(self):
        """ตั้งค่าสีพื้นหลังเป็นสีดำ"""
        self.bg_color = QColor(0, 0, 0)
        self.update_bg_color_button()
        # อัพเดตสีพื้นหลังทันที
        self.view.set_background_color(self.bg_color)

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