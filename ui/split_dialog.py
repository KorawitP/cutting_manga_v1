from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout

from ui.split_workspace import ManualSplitWorkspace


class ManualSplitDialog(QDialog):
    def __init__(self, parent=None, pixmap=None, initial_h_lines=None):
        super().__init__(parent)
        self.setWindowTitle("ตัวตัดภาพด้วยตนเอง")
        self.setMinimumSize(800, 600)

        self.workspace = ManualSplitWorkspace(self, pixmap=pixmap, initial_h_lines=initial_h_lines)
        self.workspace.accepted.connect(self.accept)
        self.workspace.cancelled.connect(self.reject)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.workspace)
        self.setLayout(layout)

        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.9), int(screen.height() * 0.9))

    def get_scale_factor(self):
        return self.workspace.get_scale_factor()

    def get_horizontal_breakpoints(self):
        return self.workspace.get_horizontal_breakpoints()

    def get_vertical_breakpoints(self):
        return self.workspace.get_vertical_breakpoints()

    def keyPressEvent(self, event):
        self.workspace.keyPressEvent(event)
