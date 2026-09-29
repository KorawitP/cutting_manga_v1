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
