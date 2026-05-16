import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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


if __name__ == "__main__":
    unittest.main()
