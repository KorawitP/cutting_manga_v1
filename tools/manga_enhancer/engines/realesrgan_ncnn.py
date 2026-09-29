import os
import shutil
import subprocess
from pathlib import Path


ENGINE_EXE = "realesrgan-ncnn-vulkan.exe" if os.name == "nt" else "realesrgan-ncnn-vulkan"
DEFAULT_MODEL = "realesr-animevideov3"


class EngineNotFoundError(RuntimeError):
    pass


def find_engine_executable(explicit_path=None):
    if explicit_path:
        exe_path = Path(explicit_path)
        if exe_path.is_file():
            return exe_path
        raise EngineNotFoundError(f"ไม่พบ engine ที่ระบุ: {exe_path}")

    tool_root = Path(__file__).resolve().parents[1]
    bundled = tool_root / "bin" / ENGINE_EXE
    if bundled.is_file():
        return bundled

    from_path = shutil.which(ENGINE_EXE)
    if from_path:
        return Path(from_path)

    raise EngineNotFoundError(
        "ไม่พบ realesrgan-ncnn-vulkan.exe ให้ดาวน์โหลด/วางไว้ที่ "
        f"{bundled} หรือเพิ่มไว้ใน PATH"
    )


def build_command(
    engine_path,
    input_path,
    output_path,
    scale=2,
    model=DEFAULT_MODEL,
    tile_size=256,
    extra_args=None,
):
    command = [
        str(engine_path),
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-n",
        model,
        "-s",
        str(scale),
        "-t",
        str(tile_size),
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def enhance_image(
    engine_path,
    input_path,
    output_path,
    scale=2,
    model=DEFAULT_MODEL,
    tile_size=256,
    extra_args=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_command(
        engine_path=engine_path,
        input_path=input_path,
        output_path=output_path,
        scale=scale,
        model=model,
        tile_size=tile_size,
        extra_args=extra_args,
    )
    return subprocess.run(command, capture_output=True, text=True)
