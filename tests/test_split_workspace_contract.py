import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QPushButton


def app():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication(sys.argv)
    return instance


class SplitWorkspaceContractTest(unittest.TestCase):
    def setUp(self):
        app()

    def test_workspace_exposes_existing_public_split_api(self):
        from ui.split_workspace import ManualSplitWorkspace

        pixmap = QPixmap(120, 180)
        pixmap.fill(Qt.white)
        workspace = ManualSplitWorkspace(pixmap=pixmap, initial_h_lines=[40, 100])

        self.assertTrue(hasattr(workspace, "get_horizontal_breakpoints"))
        self.assertTrue(hasattr(workspace, "get_vertical_breakpoints"))
        self.assertTrue(hasattr(workspace, "get_scale_factor"))
        self.assertTrue(hasattr(workspace, "get_hard_cases"))
        self.assertEqual(workspace.get_horizontal_breakpoints(), [40, 100])
        self.assertEqual(workspace.get_vertical_breakpoints(), [])
        self.assertEqual(workspace.get_scale_factor(), 1.0)

    def test_workspace_emits_accept_and_cancel_signals(self):
        from ui.split_workspace import ManualSplitWorkspace

        workspace = ManualSplitWorkspace()
        events = []
        workspace.accepted.connect(lambda: events.append("accepted"))
        workspace.cancelled.connect(lambda: events.append("cancelled"))

        buttons = {
            button.text(): button
            for button in workspace.findChildren(QPushButton)
            if button.text() in {"ตกลง", "ยกเลิก"}
        }

        buttons["ตกลง"].click()
        buttons["ยกเลิก"].click()

        self.assertEqual(events, ["accepted", "cancelled"])

    def test_dialog_wrapper_delegates_to_workspace(self):
        from ui.split_dialog import ManualSplitDialog
        from ui.split_workspace import ManualSplitWorkspace

        pixmap = QPixmap(120, 180)
        pixmap.fill(Qt.white)
        dialog = ManualSplitDialog(pixmap=pixmap, initial_h_lines=[25])

        self.assertIsInstance(dialog.workspace, ManualSplitWorkspace)
        self.assertIsInstance(dialog, QDialog)
        self.assertEqual(dialog.get_horizontal_breakpoints(), [25])
        self.assertEqual(dialog.get_vertical_breakpoints(), [])
        self.assertEqual(dialog.get_scale_factor(), 1.0)


if __name__ == "__main__":
    unittest.main()
