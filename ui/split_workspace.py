from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                           QHBoxLayout, QSplitter, QFrame)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, Signal
from ui.styles import (get_splitter_style, get_button_style, get_undo_redo_button_style)
from ui.split_view import ManualSplitView

class ManualSplitWorkspace(QWidget):
    accepted = Signal()
    cancelled = Signal()

    def __init__(self, parent=None, pixmap=None, initial_h_lines=None):
        super().__init__(parent)
        
        # กำหนดค่าเริ่มต้น
        self.pixmap = pixmap
        self.h_line_color = QColor(255, 0, 0, 150)  # สีแดงกึ่งโปร่งใส
        self.v_line_color = QColor(0, 0, 255, 150)  # สีน้ำเงินกึ่งโปร่งใส
        self.h_line_thickness = 15  # ความหนาของเส้นแนวนอน
        self.v_line_thickness = 8  # ความหนาของเส้นแนวตั้ง
        self.bg_color = QColor(240, 240, 240)  # สีพื้นหลังเทาอ่อน
        
        self.scale_factor = 1.0
        self.initial_ai_ys = []
        
        self.setup_ui()
        self.set_pixmap(self.pixmap, initial_h_lines=initial_h_lines)
            
        # เชื่อมต่อสัญญาณการเปลี่ยนแปลงประวัติ
        self.view.history_changed.connect(self.update_history_buttons)
        
        # อัพเดทสถานะของปุ่มครั้งแรก
        self.update_history_buttons()

    def set_pixmap(self, pixmap, initial_h_lines=None):
        """ตั้งค่าภาพและเส้น AI เริ่มต้นสำหรับ workspace ที่ฝังในหน้าอื่นได้"""
        self.pixmap = pixmap
        self.initial_ai_ys = []
        self.scale_factor = 1.0

        if not self.pixmap:
            return

        self.view.set_image(self.pixmap)
        # กำหนดสีและความหนาของเส้นให้กับ view
        self.view.set_line_properties(self.h_line_color, self.v_line_color, self.h_line_thickness, self.v_line_thickness)
        # กำหนดสีพื้นหลังให้กับ view
        self.view.set_background_color(self.bg_color)

        # เพิ่มเส้นเริ่มต้น (AI Guideline) ถ้ามี
        if initial_h_lines:
            self.initial_ai_ys = list(initial_h_lines)
            for y in initial_h_lines:
                self.view.add_horizontal_line(y, is_ai=True)
            # ล้างประวัติเพื่อไม่ให้ผู้ใช้กด Undo แล้วเผลอลบเส้น AI รวดเดียว
            self.view.action_history = []
            self.view.current_action_index = -1

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
        controls_container.setMinimumHeight(70)  # กำหนดความสูงขั้นต่ำของส่วนควบคุม
        controls_container.setMaximumHeight(140)  # จำกัดความสูงของส่วนควบคุม
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
            "Shift+คลิก = เส้นแนวนอน · Ctrl+Shift+คลิก = แนวตั้ง\nDelete = ลบเส้น · Ctrl+Z/Ctrl+Y = ยกเลิก/ทำซ้ำ · Ctrl+เมาส์ = ซูม"
        )
        instruction_label.setStyleSheet("color: #444; font-size: 11px;")
        instruction_label.setWordWrap(True)
        top_bar_layout.addWidget(instruction_label, 1)
        
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
        
        # แถบปุ่มด้านล่าง
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(10, 5, 10, 5)
        button_layout.addStretch()
        
        cancel_btn = QPushButton("ยกเลิก")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(self.cancelled.emit)
        cancel_btn.setStyleSheet(get_button_style("#e0e0e0", "#d0d0d0", "#c0c0c0"))
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("ตกลง")
        ok_btn.setCursor(Qt.PointingHandCursor)
        ok_btn.clicked.connect(self.accepted.emit)
        ok_btn.setStyleSheet(get_button_style("#2196F3", "#1976D2", "#0D47A1"))
        button_layout.addWidget(ok_btn)
        
        controls_layout.addLayout(button_layout)
        
        # เพิ่ม widget ไปยัง splitter
        splitter.addWidget(canvas_container)
        splitter.addWidget(controls_container)
        
        # กำหนดอัตราส่วนระหว่าง canvas และควบคุม
        splitter.setSizes([650, 140])
        
        # เพิ่ม splitter ไปยัง layout หลัก
        main_layout.addWidget(splitter)
        
        self.setMinimumSize(800, 600)
        
        # กำหนด layout หลักให้กับ workspace
        self.setLayout(main_layout)

    def get_scale_factor(self):
        """รับค่าสัดส่วนการย่อขนาด"""
        return 1.0
        
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
