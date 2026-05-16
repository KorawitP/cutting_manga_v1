import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QStackedWidget


def app():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication(sys.argv)
    return instance


class MainWindowSingleWindowContractTest(unittest.TestCase):
    def setUp(self):
        app()

    def test_main_window_owns_home_and_split_pages(self):
        from ui.main_window import ManualImageSplitter

        with patch.object(ManualImageSplitter, "log_message"):
            window = ManualImageSplitter()

        self.assertIsInstance(window.centralWidget(), QStackedWidget)
        self.assertIs(window.centralWidget(), window.stack)
        self.assertIs(window.home_page, window.stack.widget(0))
        self.assertEqual(window.stack.currentWidget(), window.home_page)

    def test_split_page_methods_are_defined_for_signal_flow(self):
        from ui.main_window import ManualImageSplitter

        required_methods = [
            "show_split_workspace",
            "return_to_home_page",
            "on_split_workspace_accepted",
            "on_split_workspace_cancelled",
            "start_merge_worker",
            "on_merge_finished",
            "start_ai_prediction_worker",
            "on_ai_prediction_finished",
            "start_export_worker",
            "on_export_finished",
            "set_workflow_busy",
        ]

        for method_name in required_methods:
            self.assertTrue(hasattr(ManualImageSplitter, method_name), method_name)

    def test_show_split_workspace_embeds_workspace_in_main_stack(self):
        from ui.main_window import ManualImageSplitter
        from ui.split_workspace import ManualSplitWorkspace

        pixmap = QPixmap(120, 180)
        pixmap.fill(Qt.white)

        with patch.object(ManualImageSplitter, "log_message"):
            window = ManualImageSplitter()

        window.show_split_workspace(pixmap, initial_h_lines=[30])

        self.assertIsInstance(window.split_workspace, ManualSplitWorkspace)
        self.assertEqual(window.stack.currentWidget(), window.split_workspace)
        self.assertEqual(window.split_workspace.get_horizontal_breakpoints(), [30])

    def test_start_manual_split_dispatches_merge_worker_flow(self):
        from ui.main_window import ManualImageSplitter

        with patch.object(ManualImageSplitter, "log_message"):
            window = ManualImageSplitter()

        window.image_files = ["chapter_1.png"]
        window.output_path_edit.setText("D:/tmp/out")
        window.episode_edit.setText("10")

        with patch.object(window, "validate_save_settings", return_value=True), \
                patch.object(window, "start_merge_worker") as start_merge_worker:
            window.start_manual_split()

        start_merge_worker.assert_called_once_with("D:/tmp/out", "10")

    def test_accepting_workspace_dispatches_export_worker_flow(self):
        from ui.main_window import ManualImageSplitter

        with patch.object(ManualImageSplitter, "log_message"):
            window = ManualImageSplitter()

        window.current_merged_img = object()
        window.current_output_dir = "D:/tmp/out"
        window.current_episode_num = "10"
        window.save_dataset_checkbox.setChecked(False)
        window.split_workspace = Mock()
        window.split_workspace.get_horizontal_breakpoints.return_value = [100]
        window.split_workspace.get_vertical_breakpoints.return_value = []
        window.split_workspace.get_scale_factor.return_value = 1.0
        window.split_workspace.get_hard_cases.return_value = {"0_unsafe": [], "1_safe": []}

        with patch.object(window, "start_export_worker") as start_export_worker:
            window.on_split_workspace_accepted()

        start_export_worker.assert_called_once_with(
            window.current_merged_img,
            [100],
            [],
            1.0,
            "D:/tmp/out",
            "10",
        )

    def test_main_window_no_longer_uses_modal_dialog_event_loop_flow(self):
        source = Path("ui/main_window.py").read_text(encoding="utf-8")

        self.assertNotIn("QEventLoop", source)
        self.assertNotIn("ManualSplitDialog", source)
        self.assertNotIn("dialog.exec()", source)


if __name__ == "__main__":
    unittest.main()
