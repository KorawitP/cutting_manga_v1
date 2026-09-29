from PySide6.QtCore import QThread, Signal

from core.image_ops import SplitResult, merge_images_vertically, split_and_save_image


class ImageMergeWorker(QThread):
    progress_changed = Signal(int, int, str)
    log_message = Signal(str)
    merge_finished = Signal(object, str)

    def __init__(self, image_files, parent=None):
        super().__init__(parent)
        self.image_files = list(image_files)
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            merged_img = merge_images_vertically(
                self.image_files,
                progress_callback=self.progress_changed.emit,
                cancel_callback=lambda: self._cancel_requested,
                log_callback=self.log_message.emit,
            )
            self.merge_finished.emit(merged_img, "")
        except Exception as exc:
            self.merge_finished.emit(None, str(exc))


class AIPredictionWorker(QThread):
    progress_changed = Signal(int)
    prediction_finished = Signal(bool, object, str)

    def __init__(self, predictor, pil_image, parent=None):
        super().__init__(parent)
        self.predictor = predictor
        self.pil_image = pil_image
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            success, cut_ys, message = self.predictor.predict_cut_points(
                self.pil_image,
                progress_callback=self.progress_changed.emit,
                cancel_callback=lambda: self._cancel_requested,
            )
            self.prediction_finished.emit(success, cut_ys, message)
        except Exception as exc:
            self.prediction_finished.emit(False, [], f"การวิเคราะห์ล้มเหลว: {exc}")


class ImageExportWorker(QThread):
    progress_changed = Signal(int, int, str)
    log_message = Signal(str)
    export_finished = Signal(object)

    def __init__(
        self,
        merged_img,
        h_breakpoints,
        v_breakpoints,
        scale_factor,
        output_dir,
        episode_num,
        source_name,
        image_format,
        quality,
        parent=None,
    ):
        super().__init__(parent)
        self.merged_img = merged_img
        self.h_breakpoints = list(h_breakpoints)
        self.v_breakpoints = list(v_breakpoints)
        self.scale_factor = scale_factor
        self.output_dir = output_dir
        self.episode_num = episode_num
        self.source_name = source_name
        self.image_format = image_format
        self.quality = quality
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            result = split_and_save_image(
                merged_img=self.merged_img,
                h_breakpoints=self.h_breakpoints,
                v_breakpoints=self.v_breakpoints,
                scale_factor=self.scale_factor,
                output_dir=self.output_dir,
                episode_num=self.episode_num,
                source_name=self.source_name,
                image_format=self.image_format,
                quality=self.quality,
                progress_callback=self.progress_changed.emit,
                cancel_callback=lambda: self._cancel_requested,
                log_callback=self.log_message.emit,
            )
            self.export_finished.emit(result)
        except Exception as exc:
            self.export_finished.emit(SplitResult(False, self.output_dir, 0, str(exc)))
