from __future__ import annotations

import argparse
from pathlib import Path

from predictor import ContentFilterPredictor, FilterConfig, discover_episode_dirs


TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = TOOL_DIR / "models" / "best_content_filter.pth"
DEFAULT_DISCARD_THRESHOLD = 0.95
DEFAULT_KEEP_THRESHOLD = 0.80
DEFAULT_ACTION = "copy"
DEFAULT_EXISTING = "skip"


def clean_path(value: str) -> Path:
    return Path(clean_input(value).strip('"').strip("'"))


def clean_input(value: str) -> str:
    return value.strip().lstrip("\ufeff")


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


def ask_float(prompt: str, default: float) -> float:
    while True:
        value = clean_input(input(f"{prompt} [{default}]: "))
        if not value:
            return default
        try:
            return float(value)
        except ValueError:
            print("กรุณาใส่ตัวเลข เช่น 0.95")


def ask_episode_dirs(mode: str | None = None) -> list[Path] | None:
    if mode is None:
        mode = ask_choice(
            "\nเลือกชนิดโฟลเดอร์",
            {
                "1": "โฟลเดอร์เรื่อง มีหลายตอน",
                "2": "โฟลเดอร์ตอนเดียว",
            },
            "1",
        )

    if mode == "2":
        return [ask_path("\nใส่ path โฟลเดอร์ตอน: ")]

    root = ask_path("\nใส่ path โฟลเดอร์เรื่อง: ")
    episode_dirs = discover_episode_dirs(root)
    if not episode_dirs:
        print("ไม่พบโฟลเดอร์ตอนที่มีรูปภาพ")
        return None
    print(f"พบ {len(episode_dirs)} ตอน")
    return episode_dirs


def default_config(existing: str = DEFAULT_EXISTING) -> FilterConfig:
    return FilterConfig(
        weights_path=DEFAULT_WEIGHTS,
        discard_threshold=DEFAULT_DISCARD_THRESHOLD,
        keep_threshold=DEFAULT_KEEP_THRESHOLD,
        action=DEFAULT_ACTION,
        existing=existing,
    )


def build_config(args) -> FilterConfig:
    return FilterConfig(
        weights_path=Path(args.weights),
        discard_threshold=args.discard_threshold,
        keep_threshold=args.keep_threshold,
        action=args.action,
        existing=args.existing,
    )


def print_config(config: FilterConfig) -> None:
    print("\nค่าที่ใช้")
    print("-" * 60)
    print(f"โมเดล: {config.weights_path}")
    print(f"การจัดไฟล์: {'copy - ปลอดภัย ต้นฉบับยังอยู่' if config.action == 'copy' else 'move - ย้ายไฟล์จริง'}")
    existing_label = {
        "skip": "ข้ามตอนที่มีผลลัพธ์แล้ว",
        "overwrite": "ลบผลเก่าแล้วทำใหม่",
        "append": "เพิ่มผลใหม่โดยไม่ลบของเก่า",
    }[config.existing]
    print(f"ผลลัพธ์เดิม: {existing_label}")
    print(f"discard threshold: {config.discard_threshold}")
    print(f"keep threshold: {config.keep_threshold}")
    print("output: แต่ละตอนจะได้โฟลเดอร์ _content_filter_result")


def print_summary(summaries):
    print("\nสรุปผล")
    print("-" * 60)
    total_keep = total_discard = total_review = total_error = total_skipped = 0

    for summary in summaries:
        if summary.skipped:
            total_skipped += 1
            print(f"{summary.episode_dir.name}: skipped (มี _content_filter_result แล้ว)")
            continue

        total_keep += summary.keep
        total_discard += summary.discard_suggested
        total_review += summary.review
        total_error += summary.error
        print(
            f"{summary.episode_dir.name}: keep={summary.keep}, "
            f"discard={summary.discard_suggested}, review={summary.review}, error={summary.error}"
        )
        print(f"  -> {summary.output_dir}")

    print("-" * 60)
    print(f"รวม keep={total_keep}, discard={total_discard}, review={total_review}, error={total_error}, skipped={total_skipped}")


def run_filter_flow(config: FilterConfig, mode: str | None = None) -> None:
    episode_dirs = ask_episode_dirs(mode)
    if not episode_dirs:
        return
    print_config(config)
    if not ask_confirm("\nเริ่มทำงาน?", True):
        print("ยกเลิก")
        return
    predictor = ContentFilterPredictor(config)
    summaries = predictor.process_episodes(episode_dirs)
    print_summary(summaries)


def configure_interactive() -> FilterConfig:
    print("\nตั้งค่า Content Filter")
    existing_choice = ask_choice(
        "\nถ้ามี _content_filter_result อยู่แล้ว",
        {
            "1": "ข้ามตอนนั้น",
            "2": "ลบผลเก่าแล้วทำใหม่",
            "3": "เพิ่มผลใหม่โดยไม่ลบของเก่า",
        },
        "1",
    )
    action_choice = ask_choice(
        "\nต้องการจัดไฟล์แบบไหน",
        {
            "1": "copy ปลอดภัย ต้นฉบับยังอยู่",
            "2": "move ย้ายไฟล์จริง",
        },
        "1",
    )
    discard_threshold = ask_float("\ndiscard threshold", DEFAULT_DISCARD_THRESHOLD)
    keep_threshold = ask_float("keep threshold", DEFAULT_KEEP_THRESHOLD)
    return FilterConfig(
        weights_path=DEFAULT_WEIGHTS,
        discard_threshold=discard_threshold,
        keep_threshold=keep_threshold,
        action={"1": "copy", "2": "move"}[action_choice],
        existing={"1": "skip", "2": "overwrite", "3": "append"}[existing_choice],
    )


def run_interactive():
    print("=" * 60)
    print("Content Filter CLI")
    print("แยกภาพหลังตัดเป็น keep / discard_suggested / review")
    print("=" * 60)

    while True:
        menu = ask_choice(
            "\nเมนูหลัก",
            {
                "1": "เรื่องหลายตอน: ข้ามของเดิม",
                "2": "เรื่องหลายตอน: ลบผลเก่าแล้วทำใหม่",
                "3": "ตอนเดียว: ข้ามของเดิม",
                "4": "ตอนเดียว: ลบผลเก่าแล้วทำใหม่",
                "5": "ตั้งค่าก่อนทำงาน",
                "6": "ดูค่าแนะนำ",
                "0": "ออก",
            },
            "1",
        )
        if menu == "0":
            return
        if menu == "1":
            run_filter_flow(default_config(), mode="1")
            return
        if menu == "2":
            run_filter_flow(default_config(existing="overwrite"), mode="1")
            return
        if menu == "3":
            run_filter_flow(default_config(), mode="2")
            return
        if menu == "4":
            run_filter_flow(default_config(existing="overwrite"), mode="2")
            return
        if menu == "5":
            run_filter_flow(configure_interactive())
            return
        if menu == "6":
            print_config(default_config())


def run_from_args(args):
    root = Path(args.path)
    if args.mode == "episode":
        episode_dirs = [root]
    else:
        episode_dirs = discover_episode_dirs(root)
        if not episode_dirs:
            raise RuntimeError(f"No episode folders with images found: {root}")

    config = build_config(args)
    predictor = ContentFilterPredictor(config)
    summaries = predictor.process_episodes(episode_dirs)
    print_summary(summaries)


def parse_args():
    parser = argparse.ArgumentParser(description="Easy content filter CLI for manga cut images.")
    parser.add_argument("--path", type=Path, help="Episode folder or story folder.")
    parser.add_argument("--mode", choices=["episode", "story"], default="story")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--discard-threshold", type=float, default=0.95)
    parser.add_argument("--keep-threshold", type=float, default=0.80)
    parser.add_argument("--action", choices=["copy", "move"], default="copy")
    parser.add_argument("--existing", choices=["skip", "overwrite", "append"], default="skip")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.path:
        run_from_args(args)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
