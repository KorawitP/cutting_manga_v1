import os
import re
import sys 
import json
import random
import shutil
import gc
import time
from io import BytesIO
from datetime import datetime
from PIL import Image

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, 
                           QFileDialog, QLabel, QMessageBox, QHBoxLayout, 
                           QProgressDialog, QProgressBar, QGroupBox, QLineEdit, 
                           QSlider, QComboBox, QTabWidget, QStatusBar, QStyle, 
                           QApplication, QDialog, QCheckBox)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QSettings

from ui.ui_components import ModernStyle, ModernButton
from ui.split_dialog import ManualSplitDialog
from core.ai_predictor import AIPredictor

# ปิดคำเตือน decompression bomb สำหรับภาพขนาดใหญ่
Image.MAX_IMAGE_PIXELS = None

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
        
        # เพิ่มการกำหนดไฟล์ log - ใช้ path ที่แน่นอนจากตำแหน่งตัว executable
        app_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.log_dir = os.path.join(app_dir, "logs")
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
        
        main_layout.addLayout(header_layout)
        
        # แท็บสำหรับแบ่งหน้าที่
        tab_widget = QTabWidget()
        
        # แท็บหลัก
        main_tab = QWidget()
        main_tab_layout = QVBoxLayout(main_tab)
        
        # ส่วนการนำเข้าไฟล์
        import_group = QGroupBox("การนำเข้าไฟล์ภาพ")
        import_layout = QHBoxLayout(import_group)
        import_layout.setSpacing(10)  # เพิ่มระยะห่างระหว่างปุ่ม
        
        # ปุ่มเลือกไฟล์ภาพ - ครึ่งซ้าย
        self.select_files_btn = ModernButton("เลือกไฟล์ภาพ", self,
                                           self.style().standardIcon(QStyle.SP_FileDialogDetailedView),
                                           self.theme["primary"])
        self.select_files_btn.setMinimumHeight(50)  # เพิ่มความสูง
        self.select_files_btn.clicked.connect(self.select_image_files)
        import_layout.addWidget(self.select_files_btn, 1)  # stretch factor = 1 (ครึ่งหนึ่ง)
        
        # ปุ่มรีเซต - ครึ่งขวา
        self.reset_btn = ModernButton("รีเซต", self,
                                    self.style().standardIcon(QStyle.SP_DialogResetButton),
                                    self.theme["error"])
        self.reset_btn.setMinimumHeight(50)  # เพิ่มความสูง
        self.reset_btn.clicked.connect(self.reset_files)
        import_layout.addWidget(self.reset_btn, 1)  # stretch factor = 1 (ครึ่งหนึ่ง)
        
        main_tab_layout.addWidget(import_group)
        
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
        self.format_combo.addItems(["JPEG", "PNG", "WebP"])
        current_format = self.settings.value("image_format", "JPEG")
        self.format_combo.setCurrentText(current_format)
        self.format_combo.currentTextChanged.connect(lambda t: self.settings.setValue("image_format", t))
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        
        image_layout.addLayout(format_layout)
        
        settings_layout.addWidget(image_group)
        
        # ตั้งค่าระบบผู้ช่วย
        ai_group = QGroupBox("ระบบผู้ช่วย")
        ai_layout = QVBoxLayout(ai_group)
        
        self.use_ai_checkbox = QCheckBox("ใช้ AI ช่วยหาจุดตัดอัตโนมัติ")
        self.use_ai_checkbox.setChecked(self.settings.value("use_ai", True, type=bool))
        self.use_ai_checkbox.stateChanged.connect(lambda state: self.settings.setValue("use_ai", bool(state)))
        ai_layout.addWidget(self.use_ai_checkbox)
        
        settings_layout.addWidget(ai_group)
        
        # ตั้งค่าที่เก็บ Dataset สำหรับเทรน AI
        dataset_group = QGroupBox("การจัดเก็บข้อมูลเทรน AI (Dataset)")
        dataset_layout = QVBoxLayout(dataset_group)
        
        dataset_path_layout = QHBoxLayout()
        dataset_path_label = QLabel("โฟลเดอร์ Dataset:")
        dataset_path_layout.addWidget(dataset_path_label)
        
        self.dataset_path_edit = QLineEdit()
        # ค่าเริ่มต้นคือโฟลเดอร์รันโปรแกรม/dataset_hardcases
        default_dataset = os.path.join(os.getcwd(), "dataset_hardcases")
        current_dataset = self.settings.value("dataset_path", default_dataset)
        self.dataset_path_edit.setText(current_dataset)
        self.dataset_path_edit.setReadOnly(True)
        dataset_path_layout.addWidget(self.dataset_path_edit)
        
        self.dataset_browse_btn = ModernButton("เปลี่ยน..", self,
                                    None,
                                    self.theme["accent"])
        self.dataset_browse_btn.clicked.connect(self.browse_dataset_folder)
        dataset_path_layout.addWidget(self.dataset_browse_btn)
        
        dataset_layout.addLayout(dataset_path_layout)
        settings_layout.addWidget(dataset_group)
        
        settings_layout.addStretch()
        
        # เพิ่มแท็บเข้าไป
        tab_widget.addTab(main_tab, "หน้าหลัก")
        tab_widget.addTab(settings_tab, "ตั้งค่า")
        
        main_layout.addWidget(tab_widget)
    
    def get_resource_path(self, relative_path):
        """หาพาธของไฟล์ทรัพยากร ทั้งตอนรันโค้ดปกติและตอนเป็น EXE"""
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)

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

    def natural_sort_key(self, s):
        """สร้าง key สำหรับจัดเรียงไฟล์แบบ Natural Sort (1, 2, 10 แทนที่จะเป็น 1, 10, 2)"""
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    def select_image_files(self):
        """เลือกไฟล์ภาพที่ต้องการแบ่ง"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "เลือกไฟล์ภาพที่ต้องการแบ่ง",
            "",
            "Image Files (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tiff)"
        )
        
        if files:
            try:
                # เพิ่มไฟล์ใหม่เข้าไปในรายการ
                added_count = 0
                for file_path in files:
                    if file_path not in self.image_files:
                        self.image_files.append(file_path)
                        added_count += 1
                
                # เรียงลำดับรายการทั้งหมดด้วย Natural Sort (ช่วยให้จัดเรียงเลข 1, 2, 10 ได้ถูกต้อง)
                self.image_files.sort(key=self.natural_sort_key)
                
                if added_count > 0:
                    self.update_file_label()
                    self.log_message(f"เพิ่มไฟล์ใหม่: {added_count} ไฟล์ (รวมทั้งหมด {len(self.image_files)} ไฟล์)")
                    self.statusBar.showMessage(f"เพิ่ม {added_count} ไฟล์สำเร็จ และจัดเรียงใหม่ตามชื่อไฟล์", 3000)
                    
                    # ตั้งค่าเริ่มต้นสำหรับโฟลเดอร์ปลายทาง
                    if not self.output_path_edit.text() and self.image_files:
                        default_folder = os.path.dirname(self.image_files[0])
                        self.output_path_edit.setText(default_folder)
                        self.log_message(f"ตั้งค่าโฟลเดอร์ปลายทางเริ่มต้น: {default_folder}")
                else:
                    QMessageBox.information(self, "แจ้งเตือน", "ไฟล์ทั้งหมดมีอยู่ในรายการแล้ว")
                    
            except Exception as e:
                self.log_message(f"เกิดข้อผิดพลาดในการเลือกไฟล์: {str(e)}")
                QMessageBox.critical(self, "ผิดพลาด", f"เกิดข้อผิดพลาด: {str(e)}")

    def is_image_file(self, file_path):
        """ตรวจสอบว่าไฟล์เป็นไฟล์ภาพหรือไม่"""
        if not os.path.isfile(file_path):
            return False
        
        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff'}
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in valid_extensions

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

    def browse_dataset_folder(self):
        """เลือกโฟลเดอร์สำหรับบันทึกภาพ Hard Cases (Dataset)"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "เลือกโฟลเดอร์สำหรับบันทึก Dataset",
            self.dataset_path_edit.text()
        )
        
        if folder:
            self.dataset_path_edit.setText(folder)
            self.settings.setValue("dataset_path", folder)
            self.log_message(f"เปลี่ยนโฟลเดอร์ Dataset เป็น: {folder}")

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
            
            # --- แทรกขั้นตอนวิเคราะห์จุดตัดด้วย AI ---
            ai_cut_ys = None
            # ใช้ get_resource_path เพื่อให้หาโมเดลเจอไม่ว่าจะอยู่ข้างนอกหรือถูกแพ็คใน EXE
            weights_path = self.get_resource_path("best_classifier.pth")
            use_ai = self.settings.value("use_ai", True, type=bool)
            
            if use_ai and os.path.exists(weights_path):
                self.log_message("พบไฟล์เบาะแส AI: กำลังรันโมเดลทำนายจุดตัด...")
                
                # แสดง Progress Dialog สำหรับ AI
                ai_progress = QProgressDialog("กำลังวิเคราะห์จุดตัดด้วย AI...", "ข้าม (Skip)", 0, 100, self)
                ai_progress.setWindowTitle("AI Guideline")
                ai_progress.setWindowModality(Qt.WindowModal)
                ai_progress.setMinimumDuration(0)
                ai_progress.setValue(0)
                
                # ฟังก์ชัน callback สำหรับให้ AI อัพเดทแถบสถานะ
                def update_ai_progress(val):
                    ai_progress.setValue(val)
                    QApplication.processEvents()
                    
                predictor = AIPredictor(weights_path)
                
                # รัน AI
                success, predicted_ys, msg = predictor.predict_cut_points(merged_img, update_ai_progress)
                
                if success:
                    ai_cut_ys = predicted_ys
                    self.log_message(f"AI ทำนายสำเร็จ: {msg}")
                else:
                    self.log_message(f"AI รันไม่สำเร็จ (ข้ามได้): {msg}")
                    
                ai_progress.close()
            else:
                if not use_ai:
                    self.log_message("ข้ามการวิเคราะห์ AI (ผู้ใช้ปิดการใช้งานในหน้าตั้งค่า)")
                else:
                    self.log_message("ข้ามการวิเคราะห์ AI (ไม่พบไฟล์ best_classifier.pth)")
            # ------------------------------------
            
            # แสดงหน้าต่างแบ่งภาพ
            self.log_message("เปิดหน้าต่างแบ่งภาพด้วยตนเอง...")
            dialog = ManualSplitDialog(self, pixmap, initial_h_lines=ai_cut_ys)
            if dialog.exec() == QDialog.Accepted:
                h_breakpoints = dialog.get_horizontal_breakpoints()
                v_breakpoints = dialog.get_vertical_breakpoints()
                scale_factor = dialog.get_scale_factor()
                hard_cases = dialog.get_hard_cases()
                if hard_cases["0_unsafe"] or hard_cases["1_safe"]:
                    self.save_hard_cases(merged_img, hard_cases)
                    
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

    def save_hard_cases(self, img, hard_cases):
        """บันทึกภาพ Hard Cases สำหรับนำไปเทรน AI ใหม่"""
        try:
            # ดึงค่าที่อยู่โฟลเดอร์จาก Settings ที่ตั้งค่าไว้ หากไม่มีใช้ค่า Default
            default_dataset = os.path.join(os.getcwd(), "dataset_hardcases")
            hardcase_dir = self.settings.value("dataset_path", default_dataset)
            
            unsafe_dir = os.path.join(hardcase_dir, "0_unsafe")
            safe_dir = os.path.join(hardcase_dir, "1_safe")
            
            os.makedirs(unsafe_dir, exist_ok=True)
            os.makedirs(safe_dir, exist_ok=True)
            
            # การตั้งค่าแบนด์วิดท์ของภาพตาม Dataset (256px height, 720px width)
            slice_height = 256
            target_width = 720
            
            # ย่อขนาดความกว้างเป็น 720 ก่อนตัด เพื่อให้ได้ภาพเหมือนกับตอน AI มอง
            scale_ratio = target_width / img.width
            new_height = int(img.height * scale_ratio)
            resized_img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # ฟังก์ชันช่วยบันทึก
            def save_slice(y_original, folder_path, prefix):
                # แปลงค่า y จาก Original เป็น 720px_scale
                y_scaled = int(y_original * scale_ratio)
                
                # ให้ค่า y อยู่กึ่งกลางแถบ 256px
                half_h = slice_height // 2
                
                top = max(0, y_scaled - half_h)
                bottom = top + slice_height
                
                # แกัสไลเดอร์ถ้าติดขอบล่าง
                if bottom > resized_img.height:
                    bottom = resized_img.height
                    top = max(0, bottom - slice_height)
                    
                # หั่นและเซฟ
                if bottom - top == slice_height:
                    slice_img = resized_img.crop((0, top, target_width, bottom))
                    filename = f"hc_{prefix}_{timestamp}_{int(y_original)}.jpg"
                    slice_img.save(os.path.join(folder_path, filename), "JPEG", quality=95)
            
            # ประมวลผลแต่ละเคส
            unsafe_count = 0
            safe_count = 0
            
            for y in hard_cases.get("0_unsafe", []):
                save_slice(y, unsafe_dir, "fp")
                unsafe_count += 1
                
            for y in hard_cases.get("1_safe", []):
                save_slice(y, safe_dir, "fn")
                safe_count += 1
                
            if unsafe_count > 0 or safe_count > 0:
                self.log_message(f"บันทึก Hard Cases สำเร็จ: Unsafe(ลบ/ย้าย)={unsafe_count} ภาพ, Safe(เพิ่ม/ย้ายมา)={safe_count} ภาพ")
                
        except Exception as e:
            self.log_message(f"เตือน: ไม่สามารถบันทึก Hard Cases ได้: {str(e)}")

    def split_at_breakpoints(self, merged_img, h_breakpoints, v_breakpoints, scale_factor=1.0, output_dir=None, episode_num=None):
        """แบ่งภาพตามจุดตัดที่กำหนด"""
        try:
            # สร้างโฟลเดอร์เก็บผลลัพธ์
            if output_dir is None:
                base_folder = os.path.dirname(self.image_files[0])
                folder_name = os.path.splitext(os.path.basename(self.image_files[0]))[0]
                output_dir = os.path.join(base_folder, f"split_results_{folder_name}")
            
            # สร้างโฟลเดอร์ย่อยใหม่ตามชื่อที่ผู้ใช้กรอก
            if episode_num:
                final_output_dir = os.path.join(output_dir, episode_num)
            else:
                folder_name = os.path.splitext(os.path.basename(self.image_files[0]))[0]
                final_output_dir = os.path.join(output_dir, folder_name)
            
            os.makedirs(final_output_dir, exist_ok=True)
            self.log_message(f"สร้างโฟลเดอร์: {final_output_dir}")
            
            # ใช้โฟลเดอร์ย่อยเป็นปลายทางสำหรับบันทึกไฟล์
            output_dir = final_output_dir
            
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

            current_format = self.format_combo.currentText()
            
            for i, img_info in enumerate(result_images):
                save_progress.setValue(i)
                save_progress.setLabelText(f"กำลังบันทึกไฟล์ {i+1}/{len(result_images)}...")
                QApplication.processEvents()  # อัพเดท UI
                
                if save_progress.wasCanceled():
                    return False
                
                # สร้างชื่อไฟล์
                ext = current_format.lower()
                if ext == 'jpeg':
                    ext = 'jpg'
                    
                if episode_num:
                    if v_breakpoints:
                        filename = f"{episode_num}_{img_info['vert_idx']}.{ext}"
                    else:
                        filename = f"{episode_num}_{img_info['vert_idx']}.{ext}"
                else:
                    folder_name = os.path.splitext(os.path.basename(self.image_files[0]))[0]
                    if v_breakpoints:
                        filename = f"{folder_name}_{img_info['vert_idx']}.{ext}"
                    else:
                        filename = f"{folder_name}_{img_info['vert_idx']}.{ext}"
                
                filepath = os.path.join(output_dir, filename)
                
                img_to_save = img_info["image"]
                
                # อ่านค่า quality จาก slider (ถ้ามี)
                quality = self.quality_slider.value() if hasattr(self, 'quality_slider') else 90
                
                if current_format == "JPEG":
                    # ถ้าภาพมี channel alpha ให้แปลงเป็น RGB ก่อนเซฟเป็น JPEG เพราะ JPEG ไม่รองรับ RGBA
                    if img_to_save.mode in ('RGBA', 'LA'):
                        background = Image.new(img_to_save.mode[:-1], img_to_save.size, (255, 255, 255))
                        background.paste(img_to_save, img_to_save.split()[-1])
                        img_to_save = background
                    img_to_save.save(filepath, current_format, quality=quality)
                elif current_format == "WebP":
                    img_to_save.save(filepath, current_format, quality=quality)
                else:
                    img_to_save.save(filepath, current_format)
                
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
