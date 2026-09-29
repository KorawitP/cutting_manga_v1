# OBS-Friendly Single Window GUI Design

## Goal

Refactor the PySide6 GUI so OBS can capture the working split screen reliably while preserving every existing user-facing feature and reducing UI freezes during long-running work.

## Current State

The application starts from `main.py` and creates `ManualImageSplitter`, a `QMainWindow` implemented in `ui/main_window.py`. The main workflow currently opens the manual split screen by constructing `ManualSplitDialog` from `ui/split_dialog.py` and running `dialog.exec()`.

The actual editing canvas lives in `ManualSplitView` from `ui/split_view.py`, which is a custom `QGraphicsView` with image display, horizontal and vertical line editing, hover behavior, drag behavior, delete, undo, redo, zoom, resize, and fit-to-screen support.

Heavy image operations already run through `QThread` workers in `ui/workers.py`:

- `ImageMergeWorker` merges selected images.
- `AIPredictionWorker` predicts horizontal cut guidelines.
- `ImageExportWorker` saves split image output.

The main risk for OBS is that the user-facing work area is in a separate modal native window. OBS window capture can remain attached to the main window while the important canvas is shown in a separate dialog handle. The main risk for UI smoothness is the nested synchronous flow around modal dialogs and local `QEventLoop.exec()` waits.

## Design Summary

Use one native top-level window for the normal workflow.

`ManualImageSplitter` becomes a shell around a `QStackedWidget` with two application pages:

- `HomePage`: the existing main controls for file selection, output path, episode name, progress, settings, AI model, and dataset options.
- `SplitWorkspace`: an embedded version of the current manual split screen containing `ManualSplitView`, undo and redo controls, split setting tabs, background controls, resize controls, and confirm/cancel buttons.

The split workspace will emit signals instead of closing a dialog:

- `accepted`
- `cancelled`
- `history_changed`, already provided by `ManualSplitView`

The existing split data API remains available:

- `get_horizontal_breakpoints()`
- `get_vertical_breakpoints()`
- `get_scale_factor()`
- `get_hard_cases()`

The main window coordinates the workflow and owns the current merged image, pixmap, worker references, progress dialogs, and cleanup.

## Architecture

### Main Window

`ManualImageSplitter` remains the root `QMainWindow`.

Responsibilities:

- Own application state and settings.
- Own the `QStackedWidget`.
- Build and show the home page.
- Build or reset the split workspace when a merged image is ready.
- Coordinate merge, AI prediction, user split editing, hard case saving, and export.
- Keep all existing validation, logging, output naming, and settings behavior.

### Home Page

The existing main tab and settings tab remain visually and behaviorally the same. The first implementation should keep the current `setup_ui()` contents as close as possible and place the existing main widget into the stacked widget as the home page.

Responsibilities:

- Select and reset image files.
- Show file details.
- Edit output folder and episode name.
- Edit image quality and format.
- Configure AI model path.
- Configure dataset hard case storage.
- Start manual split.

### Split Workspace

`ManualSplitDialog` is converted into or wrapped by a non-dialog widget named `ManualSplitWorkspace`.

Responsibilities:

- Display the merged image through `ManualSplitView`.
- Add AI guideline lines when provided.
- Preserve line settings, background controls, resize controls, undo, redo, and keyboard handling.
- Emit `accepted` when the user confirms.
- Emit `cancelled` when the user cancels.
- Ignore Escape in the same way the dialog did, unless a later approved design changes that behavior.

`ManualSplitView` should be touched only when required for embedding compatibility. Its interaction behavior should remain unchanged.

## Workflow

### Start Split

1. User clicks the existing start split button.
2. Main window validates:
   - At least one image is selected.
   - Output folder is present and writable or can be created.
   - Episode name is present and valid.
3. Main window starts `ImageMergeWorker`.
4. UI shows progress and disables duplicate start actions.
5. When merge finishes:
   - If cancelled or failed, return to idle home state.
   - If successful, convert the merged PIL image to `QPixmap`.
6. If AI is enabled and the model exists, start `AIPredictionWorker`.
7. When AI finishes or is skipped, create/reset `ManualSplitWorkspace` with the pixmap and AI guidelines.
8. Switch the stacked widget to the split workspace.

### Confirm Split

1. User clicks confirm in the split workspace.
2. Main window reads horizontal breakpoints, vertical breakpoints, scale factor, and hard cases from the workspace.
3. If hard case data exists and dataset saving is enabled, save hard cases using existing behavior.
4. If breakpoints exist, start `ImageExportWorker`.
5. UI shows export progress and disables duplicate export actions.
6. When export finishes:
   - On success, show the same success message as today.
   - Reset selected files in the same way as today.
   - Return to the home page.
   - Release merged image and pixmap references where safe.
7. If no breakpoints exist, preserve current behavior from `split_at_breakpoints`.

### Cancel Split

1. User clicks cancel in the split workspace.
2. Main window returns to the home page.
3. No export occurs.
4. Current merged image references are released.
5. File selection and save settings remain as they did before entering the split workspace.

## UI Smoothness Requirements

The UI must remain responsive during merge, AI prediction, and export.

The design preserves existing `QThread` workers and moves the main orchestration toward signal-driven callbacks instead of nested local event loops. This reduces the chance of the main UI feeling frozen while keeping the existing worker code and image logic.

Long-running operations must:

- Run outside the GUI thread.
- Update progress through Qt signals.
- Support cancellation where the current worker supports cancellation.
- Disable only actions that would corrupt the current operation.
- Re-enable UI controls on finish, cancel, or failure.
- Avoid rebuilding the entire main UI while a worker is running.

## Feature Parity Contract

The refactor must preserve these features and behaviors:

- Select multiple image files.
- Natural-sort selected image files.
- Show file count, total size, and first file name.
- Reset selected files with confirmation.
- Validate output folder.
- Create output folder when needed.
- Validate episode/file name.
- Preserve image quality setting.
- Preserve output format setting for JPEG, PNG, and WebP.
- Preserve AI enable/disable setting.
- Preserve custom AI model path and reset-to-default behavior.
- Preserve dataset hard case enable/disable setting.
- Preserve dataset folder selection.
- Merge selected images vertically.
- Preserve merge cancellation behavior.
- Convert merged PIL image to `QPixmap`.
- Run AI prediction when enabled and model exists.
- Skip AI when disabled or missing model.
- Add AI horizontal guideline lines.
- Preserve hard case detection for moved, removed, and newly added horizontal lines.
- Add horizontal lines with Shift + left click.
- Add vertical lines with Ctrl + Shift + left click.
- Drag horizontal and vertical lines.
- Delete hovered lines with Delete or Backspace.
- Undo and redo with buttons.
- Undo and redo with Ctrl+Z and Ctrl+Y.
- Preserve hover highlighting and cursor changes.
- Preserve horizontal and vertical line colors.
- Preserve horizontal and vertical line thickness controls.
- Preserve background color picker and preset background colors.
- Preserve image resize percentage control.
- Preserve fit-to-screen behavior.
- Preserve zoom with Ctrl + mouse wheel.
- Preserve vertical scrolling behavior.
- Preserve confirm and cancel actions.
- Preserve Escape protection from accidental split screen close.
- Preserve image export naming.
- Preserve vertical breakpoint crop behavior.
- Preserve horizontal split behavior.
- Preserve scale factor behavior during export.
- Preserve hard case dataset saving.
- Preserve log file creation and log messages.
- Preserve success message after export.
- Preserve file reset after successful export.

## Non-Goals

This refactor will not rewrite the image processing algorithms.

This refactor will not replace `QGraphicsView`.

This refactor will not redesign the visual style beyond the minimum necessary to embed the split workflow in the main window.

This refactor will not introduce a separate OBS-only preview window in the first implementation.

This refactor will not change keyboard shortcuts.

This refactor will not change saved output naming rules.

## Testing Strategy

Manual verification is required because the project is a desktop GUI application.

Core checks:

- Launch the app.
- Select one image and multiple images.
- Start split.
- Confirm that the app switches to the split workspace inside the same main window.
- Confirm that OBS window capture of the main window still sees the split workspace.
- Add horizontal and vertical lines.
- Drag lines.
- Delete lines.
- Use undo and redo from both buttons and keyboard shortcuts.
- Change line colors and thickness.
- Change background color.
- Resize image and use fit-to-screen.
- Confirm split and export output.
- Cancel split and confirm no export occurs.
- Run with AI enabled and a valid model.
- Run with AI disabled.
- Run with missing AI model.
- Enable dataset saving and verify hard case slices are saved when applicable.
- Cancel merge, AI prediction, and export where cancellation is available.
- Run against at least one large image set to watch for UI responsiveness.

Regression checks:

- Compare saved file names and folder structure with the previous behavior.
- Compare split counts with the same breakpoints from the previous behavior.
- Check the log file for expected major workflow messages.
- Confirm successful export still resets selected files.

## Implementation Order

1. Create `ManualSplitWorkspace` while preserving the existing split controls and APIs.
2. Keep `ManualSplitDialog` temporarily as a compatibility wrapper if useful during transition.
3. Add `QStackedWidget` to `ManualImageSplitter`.
4. Place the existing home UI in the stack.
5. Add split workspace page creation/reset logic.
6. Replace `dialog.exec()` usage with stack navigation and workspace signals.
7. Preserve current worker behavior first.
8. Convert nested `QEventLoop.exec()` orchestration to signal-driven flow.
9. Run feature parity checks.
10. Remove temporary compatibility code only after parity is verified.

## Risks And Mitigations

Risk: The embedded split workspace changes focus or shortcut behavior.

Mitigation: Keep `ManualSplitView` as the focus target when the split workspace opens and verify all shortcuts manually.

Risk: Refactoring modal flow changes cleanup timing.

Mitigation: Main window explicitly owns current merged image, pixmap, worker, and workspace references, and releases them after cancel or export finish.

Risk: Async signal flow introduces duplicate starts or exports.

Mitigation: Track operation state and disable start/confirm controls while workers are active.

Risk: OBS still fails if it captures a stale executable/window selection.

Mitigation: The primary workflow stays in one native top-level window. OBS users should capture the main application window once, not a transient dialog.

Risk: Full async conversion is too large for one pass.

Mitigation: Implement in two rounds. First embed the workspace to solve OBS. Then convert nested event loops to signal-driven orchestration for smoothness.

## Acceptance Criteria

- OBS can capture the main application window and see both the home screen and split workspace without selecting another window.
- Existing manual split features remain available.
- Existing AI, dataset, export, settings, and logging behavior remains available.
- Long-running operations still run in worker threads.
- The GUI remains responsive enough to move the window, cancel available operations, and observe progress updates during merge, AI prediction, and export.
- No intentional change is made to output files, naming, format, quality, or split coordinates.
