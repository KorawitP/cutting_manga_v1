# OBS-Friendly Single Window GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the PySide6 GUI so the full manual split workflow stays inside one main native window for OBS capture, while preserving all existing split/export/AI behavior and removing nested event-loop waits from the main workflow.

**Architecture:** Add a reusable `ManualSplitWorkspace(QWidget)` that contains the current split canvas and controls, then make `ManualSplitDialog(QDialog)` a compatibility wrapper around that widget. `ManualImageSplitter(QMainWindow)` owns a `QStackedWidget` with the existing home UI and an embedded split workspace page, and coordinates merge, optional AI prediction, split confirmation, hard-case saving, and export through worker signals.

**Tech Stack:** Python, PySide6, Pillow, existing `ui.workers` QThread classes, existing `ManualSplitView`.

---

## File Map

- Create: `tests/test_split_workspace_contract.py`
  - Contract tests for `ManualSplitWorkspace` signals, public API, AI guideline initialization, and compatibility wrapper delegation.
- Create: `tests/test_main_window_single_window_contract.py`
  - Contract tests for `ManualImageSplitter` stacked-page state without running real image workers.
- Create: `ui/split_workspace.py`
  - New embeddable widget copied from the current `ManualSplitDialog` UI and behavior.
- Modify: `ui/split_dialog.py`
  - Replace the current implementation with a thin `QDialog` wrapper around `ManualSplitWorkspace` so any old caller still works.
- Modify: `ui/main_window.py`
  - Add `QStackedWidget`, embed the split workspace, replace `dialog.exec()` with signal handlers, and convert merge/AI/export waits to callback flow.
- Modify: `docs/superpowers/specs/2026-05-16-obs-friendly-single-window-gui-design.md`
  - Add an implementation note after work is complete if behavior differs from the design, otherwise leave unchanged.

## Guardrails

- Do not modify `ui/split_view.py` unless a test proves the widget cannot be embedded without a targeted change.
- Do not change `core.image_ops`, `core.ai_predictor`, output naming, file sorting, settings keys, or dataset saving semantics.
- Keep `ManualSplitDialog` available until all internal callers have moved, so rollback and manual comparison remain easy.
- Preserve Thai UI strings from the current dialog unless a string must change because the window is no longer a dialog.
- Use the existing `ImageMergeWorker`, `AIPredictionWorker`, and `ImageExportWorker`; do not introduce new threading primitives.
- Never call `QApplication.processEvents()` as the main responsiveness strategy in the new workflow.

## Task 1: Add Split Workspace Contract Tests

**Files:**
- Create: `tests/test_split_workspace_contract.py`

- [ ] **Step 1: Write the failing contract test**

Create `tests/test_split_workspace_contract.py` with this content:

```python
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
```

- [ ] **Step 2: Run the test and verify it fails for the right reason**

Run:

```powershell
python -m unittest tests.test_split_workspace_contract -v
```

Expected: FAIL or ERROR with `ModuleNotFoundError: No module named 'ui.split_workspace'`.

## Task 2: Create `ManualSplitWorkspace`

**Files:**
- Create: `ui/split_workspace.py`
- Modify: `ui/split_dialog.py`
- Test: `tests/test_split_workspace_contract.py`

- [ ] **Step 1: Move dialog behavior into an embeddable widget**

Create `ui/split_workspace.py` by copying the current `ManualSplitDialog` implementation from `ui/split_dialog.py`, with these exact structural changes:

```python
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QSlider,
    QTabWidget, QSplitter, QFrame, QApplication, QColorDialog,
)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, Signal

from ui.styles import (
    get_splitter_style,
    get_tab_widget_style,
    get_slider_style,
    get_button_style,
    get_undo_redo_button_style,
    get_color_button_style,
)
from ui.split_view import ManualSplitView


class ManualSplitWorkspace(QWidget):
    accepted = Signal()
    cancelled = Signal()

    def __init__(self, parent=None, pixmap=None, initial_h_lines=None):
        super().__init__(parent)
        self.pixmap = pixmap
        self.h_line_color = QColor(255, 0, 0, 150)
        self.v_line_color = QColor(0, 0, 255, 150)
        self.line_thickness = 4
        self.h_line_thickness = 15
        self.v_line_thickness = 8
        self.bg_color = QColor(240, 240, 240)
        self.resize_factor = 100
        self.scale_factor = 1.0
        self.initial_ai_ys = []

        self.setup_ui()
        self.set_pixmap(self.pixmap, initial_h_lines=initial_h_lines)
        self.view.history_changed.connect(self.update_history_buttons)
        self.update_history_buttons()

    def set_pixmap(self, pixmap, initial_h_lines=None):
        self.pixmap = pixmap
        self.initial_ai_ys = []
        self.resize_slider.setValue(100)
        self.scale_factor = 1.0
        if not self.pixmap:
            return
        self.view.set_image(self.pixmap)
        self.view.set_line_properties(
            self.h_line_color,
            self.v_line_color,
            self.h_line_thickness,
            self.v_line_thickness,
        )
        self.view.set_background_color(self.bg_color)
        if initial_h_lines:
            self.initial_ai_ys = list(initial_h_lines)
            for y in initial_h_lines:
                self.view.add_horizontal_line(y, is_ai=True)
            self.view.action_history = []
            self.view.current_action_index = -1
```

In the copied `setup_ui()` method:

```python
cancel_btn.clicked.connect(self.cancelled.emit)
ok_btn.clicked.connect(self.accepted.emit)
self.setMinimumSize(800, 600)
self.setLayout(main_layout)
```

Remove dialog-only calls from the workspace:

```python
self.setWindowTitle(...)
self.resize(width, height)
self.reject
self.accept
```

Keep these methods copied unchanged except for class name references:

```python
update_resize_value
fit_to_screen
resize_image
get_scale_factor
get_horizontal_breakpoints
get_vertical_breakpoints
get_hard_cases
keyPressEvent
undo_action
redo_action
update_history_buttons
select_h_color
select_v_color
select_bg_color
set_default_bg_color
set_white_bg_color
set_black_bg_color
update_h_color_button
update_v_color_button
update_bg_color_button
change_h_thickness
change_v_thickness
apply_line_settings
```

- [ ] **Step 2: Replace `ManualSplitDialog` with a compatibility wrapper**

Replace `ui/split_dialog.py` with this wrapper shape:

```python
from PySide6.QtWidgets import QDialog, QVBoxLayout, QApplication

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

    def get_hard_cases(self):
        return self.workspace.get_hard_cases()

    def keyPressEvent(self, event):
        self.workspace.keyPressEvent(event)
```

- [ ] **Step 3: Run contract tests**

Run:

```powershell
python -m unittest tests.test_split_workspace_contract -v
```

Expected: PASS for all three tests.

- [ ] **Step 4: Run syntax verification**

Run:

```powershell
python -m py_compile ui/split_workspace.py ui/split_dialog.py
```

Expected: no output and exit code `0`.

- [ ] **Step 5: Commit the split workspace extraction**

Run:

```powershell
git add ui/split_workspace.py ui/split_dialog.py tests/test_split_workspace_contract.py
git commit -m "refactor: extract manual split workspace widget"
```

Expected: commit succeeds.

## Task 3: Add Main Window Stacked UI Contract Tests

**Files:**
- Create: `tests/test_main_window_single_window_contract.py`
- Modify later: `ui/main_window.py`

- [ ] **Step 1: Write failing main-window contract tests**

Create `tests/test_main_window_single_window_contract.py` with this content:

```python
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
```

- [ ] **Step 2: Run the test and verify it fails for the right reason**

Run:

```powershell
python -m unittest tests.test_main_window_single_window_contract -v
```

Expected: FAIL because `centralWidget()` is not a `QStackedWidget` and the new signal-flow methods do not exist yet.

## Task 4: Embed Home UI in `QStackedWidget`

**Files:**
- Modify: `ui/main_window.py`
- Test: `tests/test_main_window_single_window_contract.py`

- [ ] **Step 1: Update imports**

Change the PySide6 imports in `ui/main_window.py`:

```python
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QFileDialog, QLabel, QMessageBox,
    QHBoxLayout, QProgressDialog, QProgressBar, QGroupBox, QLineEdit,
    QSlider, QComboBox, QTabWidget, QStatusBar, QStyle, QApplication,
    QCheckBox, QStackedWidget,
)
from PySide6.QtCore import Qt, QSettings
```

Change the split import:

```python
from ui.split_workspace import ManualSplitWorkspace
```

Remove `QDialog`, `QEventLoop`, and `ManualSplitDialog` from `ui/main_window.py`.

- [ ] **Step 2: Add workflow state fields**

In `ManualImageSplitter.__init__`, after worker fields:

```python
self.progress_dialog = None
self.split_workspace = None
self.current_merged_img = None
self.current_pixmap = None
self.current_output_dir = None
self.current_episode_num = None
self.current_ai_cut_ys = None
```

- [ ] **Step 3: Wrap the existing home UI in a stack**

At the start of `setup_ui()`, replace:

```python
main_widget = QWidget()
self.setCentralWidget(main_widget)
main_layout = QVBoxLayout(main_widget)
```

with:

```python
self.stack = QStackedWidget()
self.setCentralWidget(self.stack)

self.home_page = QWidget()
self.stack.addWidget(self.home_page)
self.stack.setCurrentWidget(self.home_page)

main_widget = self.home_page
main_layout = QVBoxLayout(main_widget)
```

Keep the rest of `setup_ui()` in the home page unchanged.

- [ ] **Step 4: Add page-switch helpers**

Add these methods after `setup_ui()`:

```python
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
```

- [ ] **Step 5: Add temporary signal handler stubs**

Add these methods; later tasks will replace their bodies:

```python
def on_split_workspace_accepted(self):
    pass

def on_split_workspace_cancelled(self):
    self.log_message("ยกเลิกการแบ่งภาพด้วยตนเอง")
    self.return_to_home_page()
    gc.collect()

def start_merge_worker(self, output_dir, episode_num):
    pass

def on_merge_finished(self, merged_img, message):
    pass

def start_ai_prediction_worker(self, merged_img, weights_path):
    pass

def on_ai_prediction_finished(self, success, cut_ys, message):
    pass

def start_export_worker(self, merged_img, h_breakpoints, v_breakpoints, scale_factor, output_dir, episode_num):
    pass

def on_export_finished(self, export_result):
    pass

def set_workflow_busy(self, busy, message=None):
    self.split_btn.setEnabled(not busy)
    self.select_files_btn.setEnabled(not busy)
    self.reset_btn.setEnabled(not busy)
    self.browse_btn.setEnabled(not busy)
    if message:
        self.statusBar.showMessage(message)
```

- [ ] **Step 6: Run main-window contract tests**

Run:

```powershell
python -m unittest tests.test_main_window_single_window_contract -v
```

Expected: PASS.

- [ ] **Step 7: Commit stacked UI shell**

Run:

```powershell
git add ui/main_window.py tests/test_main_window_single_window_contract.py
git commit -m "refactor: add single-window stacked gui shell"
```

Expected: commit succeeds.

## Task 5: Replace `dialog.exec()` With Embedded Workspace Flow

**Files:**
- Modify: `ui/main_window.py`
- Test: manual smoke plus existing unit tests

- [ ] **Step 1: Update `start_manual_split()` to start workflow state**

Replace `start_manual_split()` with this signal-entry version:

```python
def start_manual_split(self):
    """เริ่มกระบวนการแบ่งภาพ"""
    if not self.image_files:
        QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเลือกไฟล์ภาพก่อน")
        return

    if not self.validate_save_settings():
        return

    self.current_output_dir = self.output_path_edit.text().strip()
    self.current_episode_num = self.episode_edit.text().strip()
    self.current_merged_img = None
    self.current_pixmap = None
    self.current_ai_cut_ys = None

    self.start_merge_worker(self.current_output_dir, self.current_episode_num)
```

- [ ] **Step 2: Implement merge worker start and finish**

Replace the temporary `start_merge_worker()` and `on_merge_finished()` bodies:

```python
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
```

- [ ] **Step 3: Implement AI worker start and finish**

Replace the temporary `start_ai_prediction_worker()` and `on_ai_prediction_finished()` bodies:

```python
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
```

- [ ] **Step 4: Implement workspace accept/cancel behavior**

Replace `on_split_workspace_accepted()` and keep `on_split_workspace_cancelled()` as:

```python
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
```

- [ ] **Step 5: Run tests and syntax verification**

Run:

```powershell
python -m unittest tests.test_split_workspace_contract tests.test_main_window_single_window_contract -v
python -m py_compile ui/main_window.py ui/split_workspace.py ui/split_dialog.py
```

Expected: tests PASS and compile produces no output.

## Task 6: Convert Export to Signal Flow

**Files:**
- Modify: `ui/main_window.py`

- [ ] **Step 1: Implement async export start**

Replace the temporary `start_export_worker()` body:

```python
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
```

- [ ] **Step 2: Implement export finish**

Replace the temporary `on_export_finished()` body:

```python
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
```

- [ ] **Step 3: Keep old synchronous helpers only if needed**

Delete `run_ai_prediction()`, `merge_all_images()`, and `split_at_breakpoints()` only after searching confirms no callers remain:

```powershell
rg -n "run_ai_prediction|merge_all_images|split_at_breakpoints" ui main.py tests
```

Expected before deletion: definitions only. Expected after deletion: no matches.

- [ ] **Step 4: Run tests and compile**

Run:

```powershell
python -m unittest tests.test_split_workspace_contract tests.test_main_window_single_window_contract -v
python -m py_compile main.py ui/main_window.py ui/split_workspace.py ui/split_dialog.py ui/workers.py
```

Expected: tests PASS and compile produces no output.

- [ ] **Step 5: Commit signal workflow**

Run:

```powershell
git add ui/main_window.py tests/test_main_window_single_window_contract.py
git commit -m "refactor: run manual split workflow in main window"
```

Expected: commit succeeds.

## Task 7: Manual Regression Checklist

**Files:**
- Modify only if a verified regression is found.

- [ ] **Step 1: Launch the app**

Run:

```powershell
python main.py
```

Expected: the existing main window opens, default page is the current home/settings UI, and no second window appears before starting split.

- [ ] **Step 2: Verify OBS-sensitive window behavior**

Manual check:

```text
1. Use OBS Window Capture on the main app window.
2. Select two or more small test images.
3. Start manual split.
4. Confirm that the canvas appears inside the same captured window.
5. Confirm no separate split dialog appears in the taskbar.
```

Expected: OBS continues showing the split canvas because the workflow remains inside the main window.

- [ ] **Step 3: Verify split workspace feature parity**

Manual check:

```text
1. Shift+left-click adds horizontal line.
2. Ctrl+Shift+left-click adds vertical line.
3. Dragging a line moves it.
4. Delete removes selected line.
5. Ctrl+Z and Ctrl+Y perform undo and redo.
6. Horizontal and vertical color buttons update line colors.
7. Horizontal and vertical thickness sliders update line thickness.
8. Background default/white/black/custom controls work.
9. Resize slider and "ปรับขนาด" change display scale.
10. "พอดีจอ" fits the image in the canvas.
11. AI guideline lines appear when AI returns cut points.
12. Hard-case dataset saving still records moved/deleted AI lines when enabled.
```

Expected: every item behaves the same as the old dialog.

- [ ] **Step 4: Verify cancel behavior**

Manual check:

```text
1. Cancel during merge progress.
2. Cancel or skip during AI progress.
3. Click "ยกเลิก" inside split workspace.
4. Cancel during export progress.
```

Expected: app remains responsive, returns to the home page after workspace cancellation, logs cancellation, and does not crash.

- [ ] **Step 5: Verify successful export**

Manual check:

```text
1. Select output folder and episode name.
2. Add at least one cut line.
3. Click "ตกลง".
4. Inspect output files.
```

Expected: output file names, image format, image quality, and reset-after-success behavior match the old workflow.

## Task 8: Final Verification and Commit

**Files:**
- Modify: `docs/superpowers/specs/2026-05-16-obs-friendly-single-window-gui-design.md` only if implementation notes are needed.

- [ ] **Step 1: Run full available automated checks**

Run:

```powershell
python -m unittest tests.test_split_workspace_contract tests.test_main_window_single_window_contract -v
python -m py_compile main.py ui/main_window.py ui/split_workspace.py ui/split_dialog.py ui/split_view.py ui/workers.py core/image_ops.py core/ai_predictor.py
```

Expected: tests PASS and compile produces no output.

- [ ] **Step 2: Run optional AI smoke test if model dependencies are installed**

Run:

```powershell
python test_ai.py
```

Expected: either all AI checks pass, or the test clearly reports missing model/dependency. Do not block the GUI refactor on missing local AI assets unless the GUI no longer imports.

- [ ] **Step 3: Inspect git diff**

Run:

```powershell
git status --short
git diff -- ui/main_window.py ui/split_workspace.py ui/split_dialog.py tests/test_split_workspace_contract.py tests/test_main_window_single_window_contract.py docs/superpowers/specs/2026-05-16-obs-friendly-single-window-gui-design.md
```

Expected: only planned files changed. Existing unrelated `tools/manga_enhancer/bin.zip` remains untracked and untouched.

- [ ] **Step 4: Final commit**

Run:

```powershell
git add ui/main_window.py ui/split_workspace.py ui/split_dialog.py tests/test_split_workspace_contract.py tests/test_main_window_single_window_contract.py docs/superpowers/specs/2026-05-16-obs-friendly-single-window-gui-design.md
git commit -m "refactor: make gui obs-friendly single window"
```

Expected: commit succeeds if there are remaining staged changes. If previous task commits already covered all changes, skip this commit and record the existing commit hashes in the final handoff.

## Rollback Plan

If the embedded workflow regresses a critical feature during manual QA:

```powershell
git revert <single-window-workflow-commit>
```

The compatibility wrapper in `ui/split_dialog.py` keeps the old dialog surface available during earlier tasks, so the lowest-risk fallback is to restore `ManualSplitDialog(self, pixmap, initial_h_lines=ai_cut_ys).exec()` while keeping `ManualSplitWorkspace` for a later attempt.

## Completion Criteria

- OBS Window Capture can follow the workflow from home page to split canvas without switching captured windows.
- No `QEventLoop.exec()` remains in `ui/main_window.py`.
- `ManualSplitView` behavior remains unchanged.
- Existing file selection, natural sorting, save settings, AI guideline, hard-case dataset, line editing, resize, fit-to-screen, export, logging, and reset behavior are preserved.
- Automated contract tests pass.
- Manual regression checklist passes on at least one real image set.
