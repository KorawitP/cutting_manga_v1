import os
from dataclasses import dataclass
from typing import Callable, Optional

from PIL import Image


ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]
LogCallback = Callable[[str], None]


@dataclass
class SplitResult:
    success: bool
    output_dir: str
    total_images: int = 0
    message: str = ""


def _is_canceled(cancel_callback: Optional[CancelCallback]) -> bool:
    return bool(cancel_callback and cancel_callback())


def merge_images_vertically(
    image_files,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_callback: Optional[CancelCallback] = None,
    log_callback: Optional[LogCallback] = None,
):
    """Merge images vertically without keeping every opened image in memory."""
    width = None
    total_height = 0
    image_infos = []

    for index, img_path in enumerate(image_files):
        if progress_callback:
            progress_callback(index, len(image_files), "กำลังตรวจสอบภาพ...")
        if _is_canceled(cancel_callback):
            return None

        try:
            with Image.open(img_path) as img:
                if width is None:
                    width = img.width
                elif img.width != width:
                    raise ValueError(f"ความกว้างของภาพไม่เท่ากัน: {img_path}")

                total_height += img.height
                image_infos.append((img_path, img.height))
        except Exception as exc:
            if log_callback:
                log_callback(f"ข้อผิดพลาดในการโหลดภาพ {img_path}: {exc}")

    if not image_infos or width is None:
        raise ValueError("ไม่สามารถโหลดภาพได้")

    if progress_callback:
        progress_callback(0, len(image_infos), "กำลังรวมภาพ...")

    merged_img = Image.new("RGB", (width, total_height), "white")
    current_height = 0

    for index, (img_path, img_height) in enumerate(image_infos):
        if progress_callback:
            progress_callback(index, len(image_infos), "กำลังรวมภาพ...")
        if _is_canceled(cancel_callback):
            return None

        with Image.open(img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            merged_img.paste(img, (0, current_height))
        current_height += img_height

    if progress_callback:
        progress_callback(len(image_infos), len(image_infos), "รวมภาพเสร็จแล้ว")

    if log_callback:
        log_callback(f"รวมภาพสำเร็จ - ขนาดภาพรวม: {width}x{total_height} pixels")

    return merged_img


def split_and_save_image(
    merged_img,
    h_breakpoints,
    v_breakpoints,
    scale_factor,
    output_dir,
    episode_num,
    source_name,
    image_format,
    quality,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_callback: Optional[CancelCallback] = None,
    log_callback: Optional[LogCallback] = None,
):
    if episode_num:
        final_output_dir = os.path.join(output_dir, episode_num)
    else:
        final_output_dir = os.path.join(output_dir, source_name)

    os.makedirs(final_output_dir, exist_ok=True)
    if log_callback:
        log_callback(f"สร้างโฟลเดอร์: {final_output_dir}")

    if scale_factor != 1.0:
        new_width = int(merged_img.width * scale_factor)
        new_height = int(merged_img.height * scale_factor)
        merged_img = merged_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        if log_callback:
            log_callback(f"ย่อขนาดภาพเป็น {scale_factor*100}% ({new_width}x{new_height})")

    if not h_breakpoints and not v_breakpoints:
        return SplitResult(True, final_output_dir, 0, "ไม่มีจุดตัดที่กำหนด ข้ามการแบ่งภาพ")

    working_image = merged_img
    if v_breakpoints:
        v_breakpoints = sorted(v_breakpoints)
        left_edge = int(v_breakpoints[0])
        right_edge = int(v_breakpoints[-1]) if len(v_breakpoints) > 1 else merged_img.width
        working_image = merged_img.crop((left_edge, 0, right_edge, merged_img.height))
        if log_callback:
            log_callback(f"ตัดขอบซ้ายที่ {left_edge}px และขอบขวาที่ {right_edge}px แล้ว")

    h_points = [0] + [int(y) for y in sorted(h_breakpoints)] + [working_image.height] if h_breakpoints else [0, working_image.height]
    total_images = len(h_points) - 1
    ext = image_format.lower()
    if ext == "jpeg":
        ext = "jpg"

    for index in range(total_images):
        if progress_callback:
            progress_callback(index, total_images, f"กำลังบันทึกไฟล์ {index + 1}/{total_images}...")
        if _is_canceled(cancel_callback):
            return SplitResult(False, final_output_dir, index, "ยกเลิกการบันทึกไฟล์")

        y_start = h_points[index]
        y_end = h_points[index + 1]
        final_crop = working_image.crop((0, y_start, working_image.width, y_end))

        filename = f"{episode_num or source_name}_{index + 1}.{ext}"
        filepath = os.path.join(final_output_dir, filename)
        img_to_save = final_crop

        if image_format == "JPEG":
            if img_to_save.mode in ("RGBA", "LA"):
                background = Image.new(img_to_save.mode[:-1], img_to_save.size, (255, 255, 255))
                background.paste(img_to_save, img_to_save.split()[-1])
                img_to_save = background
            img_to_save.save(filepath, image_format, quality=quality)
        elif image_format == "WebP":
            img_to_save.save(filepath, image_format, quality=quality)
        else:
            img_to_save.save(filepath, image_format)

        if log_callback:
            log_callback(f"บันทึกภาพ: {filepath}")

    if progress_callback:
        progress_callback(total_images, total_images, "บันทึกไฟล์เสร็จแล้ว")

    return SplitResult(True, final_output_dir, total_images)
