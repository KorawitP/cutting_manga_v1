from __future__ import annotations

import argparse
from pathlib import Path

from predictor import ContentFilterPredictor, FilterConfig, discover_episode_dirs


TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = TOOL_DIR / "models" / "best_content_filter.pth"


def clean_path(value: str) -> Path:
    return Path(value.strip().strip('"').strip("'"))


def ask_path(prompt: str) -> Path:
    while True:
        value = input(prompt).strip()
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
        marker = " (default)" if key == default else ""
        print(f"  {key}. {label}{marker}")

    while True:
        value = input("> ").strip() or default
        if value in choices:
            return value
        print("เลือกไม่ถูกต้อง ลองใหม่")


def ask_float(prompt: str, default: float) -> float:
    while True:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            return default
        try:
            return float(value)
        except ValueError:
            print("กรุณาใส่ตัวเลข เช่น 0.95")


def build_config(args) -> FilterConfig:
    return FilterConfig(
        weights_path=Path(args.weights),
        discard_threshold=args.discard_threshold,
        keep_threshold=args.keep_threshold,
        action=args.action,
        existing=args.existing,
    )


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


def run_interactive():
    print("=" * 60)
    print("Content Filter CLI")
    print("แยกภาพหลังตัดเป็น keep / discard_suggested / review")
    print("=" * 60)

    mode = ask_choice(
        "\nต้องการประมวลผลแบบไหน?",
        {
            "1": "โฟลเดอร์ตอนเดียว",
            "2": "โฟลเดอร์เรื่องที่มีหลายตอน",
        },
        "2",
    )

    if mode == "1":
        root = ask_path("\nใส่ path โฟลเดอร์ตอน: ")
        episode_dirs = [root]
    else:
        root = ask_path("\nใส่ path โฟลเดอร์เรื่อง: ")
        episode_dirs = discover_episode_dirs(root)
        if not episode_dirs:
            print("ไม่พบโฟลเดอร์ตอนที่มีรูปภาพ")
            return
        print(f"พบ {len(episode_dirs)} ตอน")

    existing = ask_choice(
        "\nถ้ามี _content_filter_result อยู่แล้วให้ทำอย่างไร?",
        {
            "skip": "ข้ามตอนนั้น",
            "overwrite": "ลบผลเก่าแล้วทำใหม่",
            "append": "เพิ่มผลใหม่โดยไม่ลบของเก่า",
        },
        "skip",
    )

    action = ask_choice(
        "\nต้องการ copy หรือ move ไฟล์?",
        {
            "copy": "copy ปลอดภัย ต้นฉบับยังอยู่",
            "move": "move ย้ายไฟล์จริง",
        },
        "copy",
    )

    discard_threshold = ask_float("\ndiscard threshold", 0.95)
    keep_threshold = ask_float("keep threshold", 0.80)

    config = FilterConfig(
        weights_path=DEFAULT_WEIGHTS,
        discard_threshold=discard_threshold,
        keep_threshold=keep_threshold,
        action=action,
        existing=existing,
    )

    print("\nตั้งค่า")
    print(f"โมเดล: {config.weights_path}")
    print(f"action: {config.action}")
    print(f"existing: {config.existing}")
    print(f"discard_threshold: {config.discard_threshold}")
    print(f"keep_threshold: {config.keep_threshold}")
    print(f"output: แต่ละตอนจะได้โฟลเดอร์ _content_filter_result")

    confirm = input("\nเริ่มทำงาน? [Y/n]: ").strip().lower()
    if confirm not in ("", "y", "yes"):
        print("ยกเลิก")
        return

    predictor = ContentFilterPredictor(config)
    summaries = predictor.process_episodes(episode_dirs)
    print_summary(summaries)


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
