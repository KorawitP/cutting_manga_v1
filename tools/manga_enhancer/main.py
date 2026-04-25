import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from engines.realesrgan_ncnn import (
    DEFAULT_MODEL,
    EngineNotFoundError,
    build_command,
    enhance_image,
    find_engine_executable,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_OUTPUT_DIR_NAME = "upscaled"
STATE_FILE_NAME = ".manga_enhancer_done.json"
TEMP_SUFFIX = ".upscaling_tmp"


@dataclass
class Job:
    root: Path
    input_file: Path
    output_file: Path


def natural_sort_key(path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", path.name)]


def is_image_file(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def format_duration(seconds):
    seconds = int(max(0, seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def make_temp_output_path(output_path):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}{TEMP_SUFFIX}{output_path.suffix}")


def state_file_for_root(root):
    root = Path(root)
    if root.is_file():
        return root.parent / STATE_FILE_NAME
    return root / STATE_FILE_NAME


def job_key(root, input_file):
    try:
        return str(input_file.relative_to(root if root.is_dir() else root.parent))
    except ValueError:
        return str(input_file)


def file_signature(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_state(state_path):
    if not state_path.exists():
        return {"done": {}}
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and isinstance(data.get("done"), dict):
            return data
    except Exception:
        pass
    return {"done": {}}


def save_state(state_path, state):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
    temp_path.replace(state_path)


def read_list_file(list_path):
    paths = []
    with Path(list_path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip().lstrip("\ufeff")
            if not stripped or stripped.startswith("#"):
                continue
            paths.append(stripped.strip('"'))
    return paths


def collect_jobs_for_root(root, output_dir_name, recursive, overwrite, keep_originals):
    root = Path(root).expanduser().resolve()
    if root.is_file():
        if not is_image_file(root):
            raise ValueError(f"ไฟล์นี้ไม่ใช่รูปภาพที่รองรับ: {root}")

        output_file = root
        if keep_originals:
            output_file = root.parent / output_dir_name / root.name
            if output_file.exists() and not overwrite:
                return root, []
        return root, [Job(root=root, input_file=root, output_file=output_file)]

    if not root.is_dir():
        raise ValueError(f"ไม่พบ path: {root}")

    output_root = root / output_dir_name
    pattern = "**/*" if recursive else "*"
    jobs = []

    for image_path in sorted(root.glob(pattern), key=natural_sort_key):
        if not is_image_file(image_path):
            continue
        if output_root in image_path.parents:
            continue
        if image_path.name.endswith(TEMP_SUFFIX + image_path.suffix):
            continue

        if keep_originals:
            if recursive:
                output_file = output_root / image_path.relative_to(root)
            else:
                output_file = output_root / image_path.name
            if output_file.exists() and not overwrite:
                continue
        else:
            output_file = image_path

        jobs.append(Job(root=root, input_file=image_path, output_file=output_file))

    return root, jobs


def apply_resume_filter(root, jobs, resume):
    if not resume:
        return jobs, load_state(state_file_for_root(root))

    state_path = state_file_for_root(root)
    state = load_state(state_path)
    done = state.get("done", {})
    pending = []

    for job in jobs:
        key = job_key(root, job.input_file)
        entry = done.get(key)
        if entry and entry.get("input") == file_signature(job.input_file):
            continue
        pending.append(job)

    return pending, state


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Upscale anime/manga images in-place by default.",
    )
    parser.add_argument("paths", nargs="*", help="ไฟล์ภาพหรือโฟลเดอร์ภาพที่ต้องการ upscale")
    parser.add_argument("--list", dest="list_file", help="ไฟล์ text ที่เก็บ path ทีละบรรทัด")
    parser.add_argument("--scale", type=int, default=2, choices=(2, 3, 4), help="อัตราขยายภาพ")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ชื่อ model ของ realesrgan-ncnn-vulkan")
    parser.add_argument("--tile", type=int, default=256, help="tile size สำหรับคุม VRAM")
    parser.add_argument("--output-dir-name", default=DEFAULT_OUTPUT_DIR_NAME, help="ชื่อโฟลเดอร์ผลลัพธ์เมื่อใช้ --keep-originals")
    parser.add_argument("--engine", default=None, help="path ไปยัง realesrgan-ncnn-vulkan.exe")
    parser.add_argument("--overwrite", action="store_true", help="เขียนทับไฟล์ในโฟลเดอร์ output เมื่อใช้ --keep-originals")
    parser.add_argument("--keep-originals", action="store_true", help="ไม่ทับไฟล์เดิม และเขียนผลลัพธ์ลง upscaled/")
    parser.add_argument("--recursive", action="store_true", help="อ่านภาพในโฟลเดอร์ย่อยด้วย")
    parser.add_argument("--resume", action="store_true", help="ข้ามไฟล์ที่เคยทำสำเร็จแล้วจาก state file")
    parser.add_argument("--dry-run", action="store_true", help="แสดงรายการที่จะทำ แต่ไม่ upscale จริง")
    return parser.parse_args(argv)


def resolve_roots(args):
    roots = list(args.paths)
    if args.list_file:
        roots.extend(read_list_file(args.list_file))
    if not roots:
        raise ValueError("กรุณาระบุ path อย่างน้อย 1 รายการ หรือใช้ --list")
    return [Path(root).expanduser().resolve() for root in roots]


def print_progress(done, total, current_name, start_time, average_seconds):
    elapsed = time.perf_counter() - start_time
    remaining = total - done
    eta = remaining * average_seconds if average_seconds else 0
    text = (
        f"\r  [{done}/{total}] {current_name:<36.36} "
        f"elapsed {format_duration(elapsed):>8} | eta {format_duration(eta):>8}"
    )
    print(text, end="", flush=True)


def run_dry_run(args, roots_jobs):
    total = sum(len(jobs) for _, jobs, _ in roots_jobs)
    print(f"พบรูปภาพ {total} ไฟล์")
    print(f"model={args.model} scale={args.scale} tile={args.tile} mode={'keep-originals' if args.keep_originals else 'in-place'}")

    for root_index, (root, jobs, _) in enumerate(roots_jobs, start=1):
        print(f"\n[Queue {root_index}/{len(roots_jobs)}] {root}")
        print(f"  Images: {len(jobs)}")
        for job in jobs:
            temp_output = make_temp_output_path(job.output_file) if job.output_file == job.input_file else job.output_file
            command = build_command(
                engine_path=args.engine or "realesrgan-ncnn-vulkan",
                input_path=job.input_file,
                output_path=temp_output,
                scale=args.scale,
                model=args.model,
                tile_size=args.tile,
            )
            action = "replace" if job.output_file == job.input_file else "write"
            print(f"  [DRY] {job.input_file} -> {job.output_file} ({action})")
            print("        " + " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command))


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])

    try:
        roots = resolve_roots(args)
        roots_jobs = []
        for root in roots:
            resolved_root, jobs = collect_jobs_for_root(
                root=root,
                output_dir_name=args.output_dir_name,
                recursive=args.recursive,
                overwrite=args.overwrite,
                keep_originals=args.keep_originals,
            )
            jobs, state = apply_resume_filter(resolved_root, jobs, args.resume)
            roots_jobs.append((resolved_root, jobs, state))
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    total_jobs = sum(len(jobs) for _, jobs, _ in roots_jobs)
    if total_jobs == 0:
        print("ไม่พบไฟล์ที่ต้อง upscale หรือถูกข้ามทั้งหมด")
        return 0

    if args.dry_run:
        run_dry_run(args, roots_jobs)
        return 0

    try:
        engine_path = find_engine_executable(args.engine)
    except EngineNotFoundError as exc:
        print(f"[ERROR] {exc}")
        print("ลองรันแบบตรวจรายการก่อนได้ด้วย --dry-run")
        return 2

    print("Manga Enhancer")
    print(f"model={args.model} scale={args.scale} tile={args.tile} mode={'keep-originals' if args.keep_originals else 'in-place'}")
    print(f"engine={engine_path}")
    print(f"queues={len(roots_jobs)} images={total_jobs}")

    global_start = time.perf_counter()
    total_done = 0
    total_failed = []
    total_success = 0

    for root_index, (root, jobs, state) in enumerate(roots_jobs, start=1):
        print(f"\n[Queue {root_index}/{len(roots_jobs)}] {root}")
        print(f"  Images: {len(jobs)}")
        folder_start = time.perf_counter()
        durations = []
        folder_success = 0
        folder_failed = []

        for index, job in enumerate(jobs, start=1):
            in_place = job.output_file == job.input_file
            engine_output = make_temp_output_path(job.output_file) if in_place else job.output_file
            print_progress(index - 1, len(jobs), job.input_file.name, folder_start, sum(durations) / len(durations) if durations else 0)

            start = time.perf_counter()
            result = enhance_image(
                engine_path=engine_path,
                input_path=job.input_file,
                output_path=engine_output,
                scale=args.scale,
                model=args.model,
                tile_size=args.tile,
            )
            elapsed = time.perf_counter() - start
            durations.append(elapsed)

            if result.returncode == 0:
                if in_place:
                    engine_output.replace(job.output_file)
                folder_success += 1
                total_success += 1
                state.setdefault("done", {})[job_key(root, job.input_file)] = {
                    "input": file_signature(job.input_file),
                    "output": str(job.output_file),
                    "scale": args.scale,
                    "model": args.model,
                    "completed_at": int(time.time()),
                }
                save_state(state_file_for_root(root), state)
            else:
                if in_place and engine_output.exists():
                    engine_output.unlink()
                folder_failed.append((job.input_file, result.stderr.strip() or result.stdout.strip()))
                total_failed.append((job.input_file, result.stderr.strip() or result.stdout.strip()))
                print()
                print(f"  FAIL {job.input_file} ({elapsed:.1f}s)")

            total_done += 1
            print_progress(index, len(jobs), job.input_file.name, folder_start, sum(durations) / len(durations))

        print()
        print(
            f"  Done: {folder_success}/{len(jobs)} images, "
            f"failed {len(folder_failed)}, elapsed {format_duration(time.perf_counter() - folder_start)}"
        )

    print()
    print(
        f"Done: {total_success}/{total_jobs} images, "
        f"failed {len(total_failed)}, total {format_duration(time.perf_counter() - global_start)}"
    )
    if total_failed:
        print("Failed files:")
        for path, error in total_failed:
            print(f"  {path}")
            if error:
                print(f"    {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
