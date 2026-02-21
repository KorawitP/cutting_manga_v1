from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsLineItem
from PySide6.QtGui import QPainter, QPen, QColor, QCursor, QBrush
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from core.models import LineAction

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
