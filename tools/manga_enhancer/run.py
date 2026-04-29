from __future__ import annotations

import argparse
import shutil
from argparse import Namespace
from pathlib import Path

from engines.realesrgan_ncnn import DEFAULT_MODEL
from main import (
    DEFAULT_OUTPUT_DIR_NAME,
    Job,
    apply_resume_filter,
    collect_jobs_for_root,
    is_image_file,
    natural_sort_key,
    run_prepared_jobs,
)


CONTENT_FILTER_RESULT_DIR = "_content_filter_result"
CONTENT_FILTER_KEEP_DIR = "keep"
DEFAULT_KEEP_OUTPUT_DIR_NAME = "keep_enhanced"
DEFAULT_SCALE = 2
DEFAULT_TILE = 256
DEFAULT_EXISTING = "skip"


def clean_path(value: str) -> Path:
    return Path(clean_input(value).strip('"').strip("'")).expanduser().resolve()


def clean_input(value: str) -> str:
    return value.strip().lstrip("\ufeff")


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        [path for path in folder.iterdir() if is_image_file(path)],
        key=natural_sort_key,
    )


def ask_path(prompt: str) -> Path:
    while True:
        value = clean_input(input(prompt))
        if not value:
            print("กรุณาใส่ path")
            continue
        path = clean_path(value)
        if path.exists():
            return path
        print(f"ไม่พบ path: {path}")


def ask_choice(prompt: str, choices: dict[str, str], default: str) -> str:
    print(prompt)
    for key, label in choices.items():
        marker = " [ค่าแนะนำ]" if key == default else ""
        print(f"  {key}. {label}{marker}")

    while True:
        value = clean_input(input("> ")) or default
        if value in choices:
            return value
        print("เลือกไม่ถูกต้อง ลองใหม่")


def ask_confirm(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    value = clean_input(input(f"{prompt} {suffix}: ")).lower()
    if not value:
        return default
    if value in {"n", "no", "0", "ไม่", "ยกเลิก"}:
        return False
    if value in {"y", "yes", "1", "ใช่", "เริ่ม"} or value.endswith("y"):
        return True
    return default


def ask_int(prompt: str, default: int, choices: set[int] | None = None) -> int:
    while True:
        value = clean_input(input(f"{prompt} [{default}]: "))
        if not value:
            return default
        try:
            parsed = int(value)
        except ValueError:
            print("กรุณาใส่ตัวเลข")
            continue
        if choices and parsed not in choices:
            print(f"เลือกได้เฉพาะ: {', '.join(str(choice) for choice in sorted(choices))}")
            continue
        return parsed


def choose_path_for_mode(mode: str) -> Path:
    prompt = "\nใส่ path โฟลเดอร์ตอน: " if mode == "episode" else "\nใส่ path โฟลเดอร์เรื่อง: "
    return ask_path(prompt)


def default_namespace(source: str, mode: str, path: Path, existing: str = DEFAULT_EXISTING) -> Namespace:
    return Namespace(
        path=path,
        mode=mode,
        source=source,
        existing=existing,
        in_place=False,
        output_dir_name=DEFAULT_KEEP_OUTPUT_DIR_NAME if source == "content-filter" else DEFAULT_OUTPUT_DIR_NAME,
        scale=DEFAULT_SCALE,
        model=DEFAULT_MODEL,
        tile=DEFAULT_TILE,
        engine=None,
        dry_run=False,
    )


def discover_episode_dirs(root: Path) -> list[Path]:
    if image_files(root):
        return [root]

    direct = [
        path for path in root.iterdir()
        if path.is_dir()
        and path.name not in {CONTENT_FILTER_RESULT_DIR, DEFAULT_OUTPUT_DIR_NAME}
        and image_files(path)
    ]
    if direct:
        return sorted(direct, key=natural_sort_key)

    leaf_dirs = []
    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue
        if CONTENT_FILTER_RESULT_DIR in folder.parts:
            continue
        if image_files(folder):
            leaf_dirs.append(folder)
    return sorted(leaf_dirs, key=lambda path: str(path).lower())


def resolve_keep_dir(path: Path) -> Path | None:
    if path.name == CONTENT_FILTER_KEEP_DIR and path.parent.name == CONTENT_FILTER_RESULT_DIR:
        return path if image_files(path) else None

    keep_dir = path / CONTENT_FILTER_RESULT_DIR / CONTENT_FILTER_KEEP_DIR
    if image_files(keep_dir):
        return keep_dir

    return None


def discover_keep_dirs(root: Path) -> list[Path]:
    keep_dir = resolve_keep_dir(root)
    if keep_dir:
        return [keep_dir]

    keep_dirs = []
    for folder in root.rglob(CONTENT_FILTER_KEEP_DIR):
        if folder.parent.name != CONTENT_FILTER_RESULT_DIR:
            continue
        if image_files(folder):
            keep_dirs.append(folder)
    return sorted(keep_dirs, key=lambda path: str(path).lower())


def output_has_images(output_dir: Path) -> bool:
    return output_dir.is_dir() and bool(image_files(output_dir))


def make_runtime_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        scale=args.scale,
        model=args.model,
        tile=args.tile,
        output_dir_name=args.output_dir_name,
        engine=args.engine,
        overwrite=args.existing == "overwrite",
        keep_originals=not args.in_place,
        recursive=False,
        resume=args.existing == "resume",
        dry_run=args.dry_run,
    )


def build_safe_output_queue(
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, list[Job], dict] | None:
    if args.existing == "skip" and output_has_images(output_dir):
        print(f"skip: มีผลลัพธ์เดิมแล้ว -> {output_dir}")
        return None

    if args.existing == "overwrite" and output_dir.exists():
        shutil.rmtree(output_dir)

    jobs = [
        Job(root=input_dir, input_file=image_path, output_file=output_dir / image_path.name)
        for image_path in image_files(input_dir)
    ]
    jobs, state = apply_resume_filter(input_dir, jobs, args.existing == "resume")
    return input_dir, jobs, state


def build_content_filter_queues(args: argparse.Namespace) -> list[tuple[Path, list[Job], dict]]:
    root = clean_path(str(args.path))
    keep_dirs = [resolve_keep_dir(root)] if args.mode == "episode" else discover_keep_dirs(root)
    keep_dirs = [path for path in keep_dirs if path is not None]
    if not keep_dirs:
        raise ValueError(f"ไม่พบ {CONTENT_FILTER_RESULT_DIR}\\{CONTENT_FILTER_KEEP_DIR} ที่มีรูปภาพ: {root}")

    queues = []
    for keep_dir in keep_dirs:
        if args.in_place:
            resolved_root, jobs = collect_jobs_for_root(
                root=keep_dir,
                output_dir_name=args.output_dir_name,
                recursive=False,
                overwrite=args.existing == "overwrite",
                keep_originals=False,
            )
            jobs, state = apply_resume_filter(
                resolved_root,
                jobs,
                args.existing in {"skip", "resume"},
            )
            queues.append((resolved_root, jobs, state))
            continue

        output_dir = keep_dir.parent / args.output_dir_name
        queue = build_safe_output_queue(keep_dir, output_dir, args)
        if queue:
            queues.append(queue)

    return queues


def build_image_folder_queues(args: argparse.Namespace) -> list[tuple[Path, list[Job], dict]]:
    root = clean_path(str(args.path))
    folders = [root] if args.mode == "episode" else discover_episode_dirs(root)
    if not folders:
        raise ValueError(f"ไม่พบโฟลเดอร์ที่มีรูปภาพ: {root}")

    queues = []
    for folder in folders:
        resolved_root, jobs = collect_jobs_for_root(
            root=folder,
            output_dir_name=args.output_dir_name,
            recursive=False,
            overwrite=args.existing == "overwrite",
            keep_originals=not args.in_place,
        )
        jobs, state = apply_resume_filter(
            resolved_root,
            jobs,
            args.existing in {"skip", "resume"} if args.in_place else args.existing == "resume",
        )
        queues.append((resolved_root, jobs, state))
    return queues


def print_plan(args: argparse.Namespace, queues: list[tuple[Path, list[Job], dict]]) -> None:
    total = sum(len(jobs) for _, jobs, _ in queues)
    print("\nตั้งค่า")
    print(f"source: {args.source}")
    print(f"mode: {args.mode}")
    print(f"path: {clean_path(str(args.path))}")
    print(f"output: {'in-place' if args.in_place else args.output_dir_name}")
    print(f"existing: {args.existing}")
    print(f"model: {args.model}")
    print(f"scale: {args.scale}")
    print(f"tile: {args.tile}")
    print(f"queues: {len(queues)}")
    print(f"images: {total}")


def run_enhance_flow(args: Namespace) -> int:
    queues = build_queues(args)
    print_plan(args, queues)
    if not ask_confirm("\nเริ่มทำงาน?", True):
        print("ยกเลิก")
        return 0
    return run_prepared_jobs(make_runtime_args(args), queues)


def configure_interactive() -> Namespace:
    print("\nตั้งค่า Manga Enhancer")
    source_choice = ask_choice(
        "\nแหล่งรูปภาพ",
        {
            "1": "ผล content_filter_cli (_content_filter_result\\keep)",
            "2": "โฟลเดอร์ภาพทั่วไป",
        },
        "1",
    )
    source = {"1": "content-filter", "2": "images"}[source_choice]

    mode_choice = ask_choice(
        "\nชนิดโฟลเดอร์",
        {
            "1": "โฟลเดอร์เรื่อง มีหลายตอน",
            "2": "โฟลเดอร์ตอนเดียว",
        },
        "1",
    )
    mode = {"1": "story", "2": "episode"}[mode_choice]
    path = choose_path_for_mode(mode)

    output_choice = ask_choice(
        "\nเขียนผลลัพธ์แบบไหน",
        {
            "1": "สร้างโฟลเดอร์ผลลัพธ์ใหม่ ปลอดภัย ไม่ทับต้นฉบับ",
            "2": "ทับไฟล์เดิมแบบ in-place",
        },
        "1",
    )
    in_place = output_choice == "2"
    output_dir_name = DEFAULT_KEEP_OUTPUT_DIR_NAME if source == "content-filter" else DEFAULT_OUTPUT_DIR_NAME
    if not in_place:
        custom_output = clean_input(input(f"\nชื่อโฟลเดอร์ผลลัพธ์ [{output_dir_name}]: "))
        if custom_output:
            output_dir_name = custom_output

    existing = ask_choice(
        "\nถ้ามีผลลัพธ์เดิม/เคยทำแล้วให้ทำอย่างไร?",
        {
            "1": "ข้ามของเดิม",
            "2": "ลบผลเก่าแล้วทำใหม่",
            "3": "ข้ามไฟล์ที่ state บอกว่าทำสำเร็จแล้ว",
        },
        "1",
    )
    existing = {"1": "skip", "2": "overwrite", "3": "resume"}[existing]

    scale = ask_int("\nscale", DEFAULT_SCALE, {2, 3, 4})
    tile = ask_int("tile", DEFAULT_TILE)
    dry_run = ask_choice(
        "\nโหมดตรวจรายการก่อนทำงานจริง",
        {
            "1": "ไม่ ใช้งานจริงทันที",
            "2": "ใช่ แสดงรายการอย่างเดียว",
        },
        "1",
    ) == "2"

    return Namespace(
        path=path,
        mode=mode,
        source=source,
        existing=existing,
        in_place=in_place,
        output_dir_name=output_dir_name,
        scale=scale,
        model=DEFAULT_MODEL,
        tile=tile,
        engine=None,
        dry_run=dry_run,
    )


def print_default_settings() -> None:
    args = default_namespace("content-filter", "story", Path("."))
    print("\nค่าแนะนำ")
    print("-" * 60)
    print("source: ผล content_filter_cli (_content_filter_result\\keep)")
    print("mode: โฟลเดอร์เรื่อง มีหลายตอน")
    print(f"output: {args.output_dir_name} (ไม่ทับต้นฉบับ)")
    print(f"existing: {DEFAULT_EXISTING} (ข้ามผลลัพธ์เดิม)")
    print(f"model: {DEFAULT_MODEL}")
    print(f"scale: {DEFAULT_SCALE}")
    print(f"tile: {DEFAULT_TILE}")


def run_interactive() -> int:
    print("=" * 60)
    print("Manga Enhancer CLI")
    print("เพิ่มความชัดภาพ และทำงานต่อจาก content_filter_cli ได้")
    print("=" * 60)

    while True:
        menu = ask_choice(
            "\nเมนูหลัก",
            {
                "1": "ผล content_filter: เรื่องหลายตอน ข้ามของเดิม",
                "2": "ผล content_filter: เรื่องหลายตอน ลบผลเก่าแล้วทำใหม่",
                "3": "ภาพทั่วไป: เรื่องหลายตอน ข้ามของเดิม",
                "4": "ภาพทั่วไป: เรื่องหลายตอน ลบผลเก่าแล้วทำใหม่",
                "5": "ตอนเดียว",
                "6": "ตั้งค่าก่อนทำงาน",
                "7": "ดูค่าแนะนำ",
                "0": "ออก",
            },
            "1",
        )
        if menu == "0":
            return 0
        if menu == "1":
            return run_enhance_flow(default_namespace("content-filter", "story", choose_path_for_mode("story")))
        if menu == "2":
            return run_enhance_flow(default_namespace("content-filter", "story", choose_path_for_mode("story"), existing="overwrite"))
        if menu == "3":
            return run_enhance_flow(default_namespace("images", "story", choose_path_for_mode("story")))
        if menu == "4":
            return run_enhance_flow(default_namespace("images", "story", choose_path_for_mode("story"), existing="overwrite"))
        if menu == "5":
            source_choice = ask_choice(
                "\nแหล่งรูปภาพ",
                {
                    "1": "ผล content_filter_cli (_content_filter_result\\keep)",
                    "2": "โฟลเดอร์ภาพทั่วไป",
                },
                "1",
            )
            source = {"1": "content-filter", "2": "images"}[source_choice]
            return run_enhance_flow(default_namespace(source, "episode", choose_path_for_mode("episode")))
        if menu == "6":
            return run_enhance_flow(configure_interactive())
        if menu == "7":
            print_default_settings()


def build_queues(args: argparse.Namespace) -> list[tuple[Path, list[Job], dict]]:
    if args.source == "content-filter":
        return build_content_filter_queues(args)
    return build_image_folder_queues(args)


def run_from_args(args: argparse.Namespace) -> int:
    if not args.path:
        return run_interactive()

    queues = build_queues(args)
    print_plan(args, queues)
    return run_prepared_jobs(make_runtime_args(args), queues)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Easy manga enhancer CLI.")
    parser.add_argument("--path", type=Path, help="Episode folder, story folder, or keep folder.")
    parser.add_argument("--mode", choices=["episode", "story"], default="story")
    parser.add_argument("--source", choices=["content-filter", "images"], default="content-filter")
    parser.add_argument("--existing", choices=["skip", "overwrite", "resume"], default="skip")
    parser.add_argument("--output-dir-name", default=None)
    parser.add_argument("--in-place", action="store_true", help="Replace input images after successful upscale.")
    parser.add_argument("--scale", type=int, default=2, choices=(2, 3, 4))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--engine", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output_dir_name is None:
        args.output_dir_name = (
            DEFAULT_KEEP_OUTPUT_DIR_NAME if args.source == "content-filter" else DEFAULT_OUTPUT_DIR_NAME
        )
    return args


def main() -> int:
    try:
        return run_from_args(parse_args())
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
