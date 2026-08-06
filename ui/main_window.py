import os
import re
import sys 
import gc
from datetime import datetime
from PIL import Image

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                           QFileDialog, QLabel, QMessageBox, QHBoxLayout, 
                           QProgressDialog, QProgressBar, QGroupBox, QLineEdit, 
                           QSlider, QComboBox, QTabWidget, QStatusBar, QStyle, 
                           QCheckBox, QStackedWidget)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QSettings

from ui.ui_components import ModernStyle, ModernButton
from ui.split_workspace import ManualSplitWorkspace
from ui.workers import AIPredictionWorker, ImageExportWorker, ImageMergeWorker
from core.ai_predictor import AIPredictor
from core.image_ops import save_hard_case_slices

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
        self.ai_predictor = None
        self.ai_weights_path = None
        self.ai_worker = None
        self.merge_worker = None
        self.export_worker = None
        self.progress_dialog = None
        self.split_workspace = None
        self.current_merged_img = None
        self.current_pixmap = None
        self.current_output_dir = None
        self.current_episode_num = None
        self.current_ai_cut_ys = None
        
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
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_page = QWidget()
        self.stack.addWidget(self.home_page)
        self.stack.setCurrentWidget(self.home_page)

        main_widget = self.home_page
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

        ai_model_layout = QHBoxLayout()
        ai_model_label = QLabel("ไฟล์โมเดล AI:")
        ai_model_layout.addWidget(ai_model_label)

        self.ai_model_path_edit = QLineEdit()
        self.ai_model_path_edit.setReadOnly(True)
        self.ai_model_path_edit.setPlaceholderText("ค่าเริ่มต้น: cnn_v5.pth")
        current_ai_model = self.settings.value("ai_model_path", "")
        self.ai_model_path_edit.setText(current_ai_model)
        ai_model_layout.addWidget(self.ai_model_path_edit)

        self.ai_model_browse_btn = ModernButton("เลือก..", self, None, self.theme["accent"])
        self.ai_model_browse_btn.clicked.connect(self.browse_ai_model_file)
        ai_model_layout.addWidget(self.ai_model_browse_btn)

        self.ai_model_reset_btn = ModernButton("ค่าเริ่มต้น", self, None, self.theme["secondary"])
        self.ai_model_reset_btn.clicked.connect(self.reset_ai_model_file)
        ai_model_layout.addWidget(self.ai_model_reset_btn)

        ai_layout.addLayout(ai_model_layout)
        
        settings_layout.addWidget(ai_group)
        
        # ตั้งค่าที่เก็บ Dataset สำหรับเทรน AI
        dataset_group = QGroupBox("การจัดเก็บข้อมูลเทรน AI (Dataset)")
        dataset_layout = QVBoxLayout(dataset_group)

        self.save_dataset_checkbox = QCheckBox("เก็บ Dataset สำหรับเทรน AI")
        self.save_dataset_checkbox.setChecked(self.settings.value("save_dataset", False, type=bool))
        self.save_dataset_checkbox.stateChanged.connect(self.on_save_dataset_changed)
        dataset_layout.addWidget(self.save_dataset_checkbox)
        
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
        self.update_dataset_controls_enabled()
        settings_layout.addWidget(dataset_group)
        
        settings_layout.addStretch()
        
        # เพิ่มแท็บเข้าไป
        tab_widget.addTab(main_tab, "หน้าหลัก")
        tab_widget.addTab(settings_tab, "ตั้งค่า")
        
        main_layout.addWidget(tab_widget)

    def return_to_home_page(self):
        if hasattr(self, "stack") and hasattr(self, "home_page"):
            self.stack.setCurrentWidget(self.home_page)
        if self.split_workspace is not None:
            self.stack.removeWidget(self.split_workspace)
            self.split_workspace.deleteLater()
            self.split_workspace = None
        self.current_pixmap = None
        self.current_ai_cut_ys = None

    def show_split_workspace(self, pixmap, initial_h_lines=None):
        if self.split_workspace is not None:
            self.stack.removeWidget(self.split_workspace)
            self.split_workspace.deleteLater()

        self.split_workspace = ManualSplitWorkspace(self, pixmap=pixmap, initial_h_lines=initial_h_lines)
        self.split_workspace.accepted.connect(self.on_split_workspace_accepted)
        self.split_workspace.cancelled.connect(self.on_split_workspace_cancelled)
        self.stack.addWidget(self.split_workspace)
        self.stack.setCurrentWidget(self.split_workspace)
        self.split_workspace.view.setFocus()
        self.statusBar.showMessage("กำลังแบ่งภาพด้วยตนเอง", 3000)

    def on_split_workspace_accepted(self):
        if self.split_workspace is None or self.current_merged_img is None:
            self.return_to_home_page()
            return

        h_breakpoints = self.split_workspace.get_horizontal_breakpoints()
        v_breakpoints = self.split_workspace.get_vertical_breakpoints()
        scale_factor = self.split_workspace.get_scale_factor()
        hard_cases = self.split_workspace.get_hard_cases()

        if hard_cases["0_unsafe"] or hard_cases["1_safe"]:
            if self.save_dataset_checkbox.isChecked():
                self.save_hard_cases(self.current_merged_img, hard_cases)
            else:
                self.log_message("ข้ามการบันทึก Dataset (ผู้ใช้ไม่ได้เปิดโหมดเก็บ Dataset)")

        if h_breakpoints or v_breakpoints:
            self.start_export_worker(
                self.current_merged_img,
                h_breakpoints,
                v_breakpoints,
                scale_factor,
                self.current_output_dir,
                self.current_episode_num,
            )
            return

        self.log_message("ไม่มีจุดตัดที่ผู้ใช้กำหนด")
        self.return_to_home_page()
        gc.collect()

    def on_split_workspace_cancelled(self):
        self.log_message("ยกเลิกการแบ่งภาพด้วยตนเอง")
        self.return_to_home_page()
        self.set_workflow_busy(False, "พร้อมใช้งาน")
        self.current_merged_img = None
        gc.collect()

    def start_merge_worker(self, output_dir, episode_num):
        self.progress_dialog = QProgressDialog("กำลังเตรียมภาพ...", "ยกเลิก", 0, len(self.image_files), self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)

        self.log_message("\nกำลังรวมภาพทั้งหมดเข้าด้วยกัน...")
        self.set_workflow_busy(True, "กำลังรวมภาพ...")

        worker = ImageMergeWorker(self.image_files, self)
        self.merge_worker = worker

        def update_progress(value, maximum, label):
            if self.progress_dialog is None:
                return
            self.progress_dialog.setMaximum(maximum)
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(label)

        self.progress_dialog.canceled.connect(worker.cancel)
        worker.progress_changed.connect(update_progress)
        worker.log_message.connect(self.log_message)
        worker.merge_finished.connect(self.on_merge_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def on_merge_finished(self, merged_img, message):
        self.merge_worker = None
        if self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog = None

        if message:
            self.log_message(f"เกิดข้อผิดพลาดในการรวมภาพ: {message}")
            QMessageBox.critical(self, "ผิดพลาด", f"เกิดข้อผิดพลาดในการรวมภาพ: {message}")
            self.set_workflow_busy(False, "พร้อมใช้งาน")
            gc.collect()
            return

        if merged_img is None:
            self.log_message("ยกเลิกการรวมภาพ")
            self.set_workflow_busy(False, "พร้อมใช้งาน")
            gc.collect()
            return

        self.current_merged_img = merged_img
        self.statusBar.showMessage("กำลังเตรียมภาพสำหรับแสดงผล...")
        self.current_pixmap = self.pil_image_to_pixmap(merged_img)

        weights_path = self.get_ai_model_path()
        use_ai = self.settings.value("use_ai", True, type=bool)
        if use_ai and os.path.exists(weights_path):
            self.log_message(f"พบไฟล์โมเดล AI: {weights_path}")
            self.log_message("กำลังรันโมเดลทำนายจุดตัด...")
            self.start_ai_prediction_worker(merged_img, weights_path)
            return

        if not use_ai:
            self.log_message("ข้ามการวิเคราะห์ AI (ผู้ใช้ปิดการใช้งานในหน้าตั้งค่า)")
        else:
            self.log_message(f"ข้ามการวิเคราะห์ AI (ไม่พบไฟล์โมเดล: {weights_path})")

        self.set_workflow_busy(False, "พร้อมแบ่งภาพ")
        self.log_message("เปิดหน้าแบ่งภาพด้วยตนเอง...")
        self.show_split_workspace(self.current_pixmap, initial_h_lines=None)

    def start_ai_prediction_worker(self, merged_img, weights_path):
        if self.ai_predictor is None or self.ai_weights_path != weights_path:
            self.ai_predictor = AIPredictor(weights_path)
            self.ai_weights_path = weights_path

        self.progress_dialog = QProgressDialog("กำลังวิเคราะห์จุดตัดด้วย AI...", "ข้าม (Skip)", 0, 100, self)
        self.progress_dialog.setWindowTitle("AI Guideline")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)

        worker = AIPredictionWorker(self.ai_predictor, merged_img, self)
        self.ai_worker = worker

        self.progress_dialog.canceled.connect(worker.cancel)
        worker.progress_changed.connect(self.progress_dialog.setValue)
        worker.prediction_finished.connect(self.on_ai_prediction_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def on_ai_prediction_finished(self, success, cut_ys, message):
        self.ai_worker = None
        if self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog = None

        if success:
            self.current_ai_cut_ys = cut_ys
            self.log_message(f"AI ทำนายสำเร็จ: {message}")
        else:
            self.current_ai_cut_ys = None
            self.log_message(f"AI รันไม่สำเร็จ (ข้ามได้): {message}")

        self.set_workflow_busy(False, "พร้อมแบ่งภาพ")
        self.log_message("เปิดหน้าแบ่งภาพด้วยตนเอง...")
        self.show_split_workspace(self.current_pixmap, initial_h_lines=self.current_ai_cut_ys)

    def start_export_worker(self, merged_img, h_breakpoints, v_breakpoints, scale_factor, output_dir, episode_num):
        source_name = os.path.splitext(os.path.basename(self.image_files[0]))[0]
        total_parts = len(h_breakpoints) + 1 if h_breakpoints else 1

        self.progress_dialog = QProgressDialog("กำลังบันทึกไฟล์...", "ยกเลิก", 0, total_parts, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.set_workflow_busy(True, "กำลังบันทึกไฟล์...")

        worker = ImageExportWorker(
            merged_img,
            h_breakpoints,
            v_breakpoints,
            scale_factor,
            output_dir,
            episode_num,
            source_name,
            self.format_combo.currentText(),
            self.quality_slider.value() if hasattr(self, "quality_slider") else 90,
            self,
        )
        self.export_worker = worker

        def update_progress(value, maximum, label):
            if self.progress_dialog is None:
                return
            self.progress_dialog.setMaximum(maximum)
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(label)

        self.progress_dialog.canceled.connect(worker.cancel)
        worker.progress_changed.connect(update_progress)
        worker.log_message.connect(self.log_message)
        worker.export_finished.connect(self.on_export_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def on_export_finished(self, export_result):
        self.export_worker = None
        if self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog = None

        self.set_workflow_busy(False, "พร้อมใช้งาน")

        if export_result is None:
            self.log_message("ยกเลิกการบันทึกไฟล์")
            return

        if export_result.message:
            self.log_message(export_result.message)

        if export_result.success:
            self.log_message(
                f"แบ่งภาพเสร็จสิ้น {export_result.total_images} ส่วน บันทึกที่: {export_result.output_dir}"
            )
            QMessageBox.information(self, "สำเร็จ", "แบ่งภาพเสร็จสิ้น")
            self.return_to_home_page()
            self.image_files = []
            self.update_file_label()
            self.log_message("รีเซ็ตรายการภาพหลังจากตัดเสร็จ")
        else:
            QMessageBox.critical(self, "ผิดพลาด", export_result.message or "บันทึกไฟล์ไม่สำเร็จ")

        self.current_merged_img = None
        gc.collect()

    def set_workflow_busy(self, busy, message=None):
        self.split_btn.setEnabled(not busy)
        self.select_files_btn.setEnabled(not busy)
        self.reset_btn.setEnabled(not busy)
        self.browse_btn.setEnabled(not busy)
        if message:
            self.statusBar.showMessage(message)
    
    def get_resource_path(self, relative_path):
        """หาพาธของไฟล์ทรัพยากร ทั้งตอนรันโค้ดปกติและตอนเป็น EXE"""
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)

    def get_ai_model_path(self):
        """คืน path โมเดล AI ที่ผู้ใช้เลือก หรือ fallback เป็น cnn_v5.pth"""
        custom_path = self.settings.value("ai_model_path", "", type=str).strip()
        if custom_path:
            return custom_path
        return self.get_resource_path("cnn_v5.pth")

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

    def on_save_dataset_changed(self, state):
        """บันทึกสถานะการเก็บ Dataset และเปิด/ปิดช่องตั้งค่าที่เกี่ยวข้อง"""
        enabled = bool(state)
        self.settings.setValue("save_dataset", enabled)
        self.update_dataset_controls_enabled()
        if enabled:
            self.log_message("เปิดการเก็บ Dataset สำหรับเทรน AI")
        else:
            self.log_message("ปิดการเก็บ Dataset สำหรับเทรน AI")

    def update_dataset_controls_enabled(self):
        """เปิดปุ่มเลือกโฟลเดอร์ Dataset เฉพาะเมื่อเปิดโหมดเก็บ Dataset"""
        enabled = self.save_dataset_checkbox.isChecked()
        self.dataset_path_edit.setEnabled(enabled)
        self.dataset_browse_btn.setEnabled(enabled)

    def browse_ai_model_file(self):
        """เลือกไฟล์โมเดล AI สำหรับทำนายจุดตัด"""
        start_dir = self.ai_model_path_edit.text() or self.get_resource_path("cnn_v5.pth")
        if os.path.isfile(start_dir):
            start_dir = os.path.dirname(start_dir)

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "เลือกไฟล์โมเดล AI",
            start_dir,
            "PyTorch Model (*.pth *.pt);;All Files (*)"
        )

        if file_path:
            self.ai_model_path_edit.setText(file_path)
            self.settings.setValue("ai_model_path", file_path)
            self.ai_predictor = None
            self.ai_weights_path = None
            self.log_message(f"เปลี่ยนไฟล์โมเดล AI เป็น: {file_path}")

    def reset_ai_model_file(self):
        """กลับไปใช้โมเดลค่าเริ่มต้น cnn_v5.pth"""
        self.ai_model_path_edit.clear()
        self.settings.setValue("ai_model_path", "")
        self.ai_predictor = None
        self.ai_weights_path = None
        self.log_message("เปลี่ยนไฟล์โมเดล AI กลับเป็นค่าเริ่มต้น: cnn_v5.pth")

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

    def pil_image_to_pixmap(self, pil_image):
        """Convert a PIL image to QPixmap without PNG encode/decode overhead."""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        width, height = pil_image.size
        image_bytes = pil_image.tobytes("raw", "RGB")
        qimg = QImage(image_bytes, width, height, width * 3, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

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

        self.current_output_dir = output_dir
        self.current_episode_num = episode_num
        self.current_merged_img = None
        self.current_pixmap = None
        self.current_ai_cut_ys = None

        try:
            self.start_merge_worker(output_dir, episode_num)
        except Exception as e:
            self.log_message(f"เกิดข้อผิดพลาด: {str(e)}")
            QMessageBox.critical(self, "ผิดพลาด", f"เกิดข้อผิดพลาด: {str(e)}")
            self.set_workflow_busy(False, "พร้อมใช้งาน")
            gc.collect()

    def save_hard_cases(self, img, hard_cases):
        """บันทึกภาพ Hard Cases สำหรับนำไปเทรน AI ใหม่"""
        try:
            default_dataset = os.path.join(os.getcwd(), "dataset_hardcases")
            hardcase_dir = self.settings.value("dataset_path", default_dataset)
            save_hard_case_slices(img, hard_cases, hardcase_dir, log_callback=self.log_message)
        except Exception as e:
            self.log_message(f"เตือน: ไม่สามารถบันทึก Hard Cases ได้: {str(e)}")

