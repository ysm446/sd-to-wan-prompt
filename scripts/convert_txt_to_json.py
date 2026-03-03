"""
Convert WAN prompt text files into session JSON files.

Expected txt structure:
=== Original Prompt ===
...
=== Additional Instruction ===
...
=== Generated WAN Prompt ===
...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.image_parser import ImageParser  # noqa: E402


HEADERS = [
    "=== Original Prompt ===",
    "=== Additional Instruction ===",
    "=== Generated WAN Prompt ===",
]


def parse_txt_sections(text: str) -> Dict[str, str]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    current = None
    buckets: Dict[str, List[str]] = {
        "original_prompt": [],
        "additional_instruction": [],
        "generated_prompt": [],
    }

    for line in lines:
        stripped = line.strip()
        if stripped == HEADERS[0]:
            current = "original_prompt"
            continue
        if stripped == HEADERS[1]:
            current = "additional_instruction"
            continue
        if stripped == HEADERS[2]:
            current = "generated_prompt"
            continue
        if current is not None:
            buckets[current].append(line)

    return {k: "\n".join(v).strip() for k, v in buckets.items()}


def is_target_txt(text: str) -> bool:
    return HEADERS[0] in text and HEADERS[2] in text


def build_session_payload(txt_path: Path) -> Dict:
    sections = parse_txt_sections(txt_path.read_text(encoding="utf-8"))
    image_path = txt_path.with_suffix(".png")

    metadata = {
        "path": str(image_path),
        "filename": image_path.name,
        "size": None,
        "prompt": sections["original_prompt"],
        "negative_prompt": "",
        "settings": {},
    }
    if image_path.exists():
        try:
            metadata = ImageParser.extract_metadata(str(image_path))
            if not metadata.get("prompt"):
                metadata["prompt"] = sections["original_prompt"]
        except Exception:
            # Keep fallback metadata if extraction fails.
            pass

    return {
        "image_filename": image_path.name,
        "image_path": str(image_path),
        "metadata": metadata,
        "prompt": sections["generated_prompt"],
        "additional_instruction": sections["additional_instruction"],
        "original_prompt": sections["original_prompt"],
    }


def convert_folder(target_dir: Path, recursive: bool, overwrite: bool) -> int:
    pattern = "**/*.txt" if recursive else "*.txt"
    txt_files = sorted(target_dir.glob(pattern))
    converted = 0

    for txt_path in txt_files:
        raw_text = txt_path.read_text(encoding="utf-8")
        if not is_target_txt(raw_text):
            print(f"[SKIP] {txt_path.name} (not WAN txt format)")
            continue

        json_path = txt_path.with_suffix(".json")
        if json_path.exists() and not overwrite:
            print(f"[SKIP] {json_path} already exists")
            continue

        payload = build_session_payload(txt_path)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        converted += 1
        print(f"[OK] {txt_path.name} -> {json_path.name}")

    print(f"\nConverted: {converted} file(s)")
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert WAN txt files to session json")
    parser.add_argument(
        "target_dir",
        nargs="?",
        default=".",
        help="Target folder containing txt files (default: current directory)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process txt files recursively",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing json files",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"[ERROR] Folder not found: {target_dir}")
        return 1

    convert_folder(target_dir, recursive=args.recursive, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
