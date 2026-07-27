#!/usr/bin/env python3
"""Batch-resize images to 2048x2048.

Default input:  results/swinir_real_sr_x2
Default output: results/swinir_real_sr_x2_resize512
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def resize_folder(
    input_dir: Path,
    output_dir: Path,
    size: int,
    overwrite: bool,
) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]

    if not image_paths:
        print(f"No supported images found in: {input_dir}")
        return

    processed = 0
    skipped_exists = 0

    for src in image_paths:
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not overwrite:
            skipped_exists += 1
            continue

        with Image.open(src) as img:
            resized = img.resize((size, size), Image.LANCZOS)
            resized.save(dst)
            processed += 1

    print("Done.")
    print(f"Processed: {processed}")
    print(f"Skipped (exists): {skipped_exists}")
    print(f"Output directory: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch resize images to square size (default: 2048x2048).")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data\TEM\zwl"),
        help="Input folder containing images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data\TEM\zwl_resize512"),
        help="Output folder for resized images.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Output size (square).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resize_folder(args.input, args.output, args.size, args.overwrite)


if __name__ == "__main__":
    main()
