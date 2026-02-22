from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                           QHBoxLayout, QDialog, QSlider, QTabWidget, 
                           QSplitter, QFrame, QApplication, QColorDialog)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt
from ui.styles import (get_splitter_style, get_tab_widget_style, get_slider_style, 
                   get_button_style, get_undo_redo_button_style, get_color_button_style)
from ui.split_view import ManualSplitView

class ManualSplitDialog(QDialog):
    def __init__(self, parent=None, pixmap=None, initial_h_lines=None):
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
            
            # เพิ่มเส้นเริ่มต้น (AI Guideline) ถ้ามี
            if initial_h_lines:
                for y in initial_h_lines:
                    self.view.add_horizontal_line(y)
                # ล้างประวัติเพื่อไม่ให้ผู้ใช้กด Undo แล้วเผลอลบเส้น AI รวดเดียว
                self.view.action_history = []
                self.view.current_action_index = -1
            
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
        splitter.setStyleSheet(get_splitter_style())
        
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
        self.undo_btn.setStyleSheet(get_undo_redo_button_style())
        top_bar_layout.addWidget(self.undo_btn)
        
        self.redo_btn = QPushButton("↪")
        self.redo_btn.setToolTip("ทำซ้ำ (Redo)")
        self.redo_btn.setFixedSize(28, 28)
        self.redo_btn.clicked.connect(self.redo_action)
        self.redo_btn.setStyleSheet(get_undo_redo_button_style())
        top_bar_layout.addWidget(self.redo_btn)
        
        controls_layout.addWidget(top_bar)
        
        # แท็บควบคุม - ใช้ QTabWidget ที่มี style แบบ modern
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)
        tabs.setStyleSheet(get_tab_widget_style())
        
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
        self.h_thickness_slider.setStyleSheet(get_slider_style())
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
        self.v_thickness_slider.setStyleSheet(get_slider_style())
        self.v_thickness_slider.valueChanged.connect(self.change_v_thickness)
        line_layout.addWidget(self.v_thickness_slider)
        
        self.v_thickness_value = QLabel(f"{self.v_line_thickness}px")
        self.v_thickness_value.setStyleSheet("color: #444444; min-width: 35px;")
        line_layout.addWidget(self.v_thickness_value)
        
        # Apply ปุ่ม
        self.apply_line_settings_btn = QPushButton("ปรับใช้")
        self.apply_line_settings_btn.setCursor(Qt.PointingHandCursor)
        self.apply_line_settings_btn.clicked.connect(self.apply_line_settings)
        self.apply_line_settings_btn.setStyleSheet(get_button_style("#4CAF50", "#45a049", "#3d8b40"))
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
        color_btn_style = get_color_button_style()
        
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
        self.resize_slider.setStyleSheet(get_slider_style())
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
        self.apply_resize_btn.setStyleSheet(get_button_style("#2196F3", "#1976D2", "#0D47A1"))
        resize_layout.addWidget(self.apply_resize_btn)
        
        # ปุ่มพอดีกับหน้าจอ (fit to screen)
        self.fit_screen_btn = QPushButton("พอดีจอ")
        self.fit_screen_btn.setCursor(Qt.PointingHandCursor)
        self.fit_screen_btn.clicked.connect(self.fit_to_screen)
        self.fit_screen_btn.setStyleSheet(get_button_style("#FF9800", "#F57C00", "#E65100"))
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
        cancel_btn.setStyleSheet(get_button_style("#e0e0e0", "#d0d0d0", "#c0c0c0"))
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("ตกลง")
        ok_btn.setCursor(Qt.PointingHandCursor)
        ok_btn.clicked.connect(self.accept)
        ok_btn.setStyleSheet(get_button_style("#2196F3", "#1976D2", "#0D47A1"))
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
