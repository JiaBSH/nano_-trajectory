#!/usr/bin/env python
"""Detect fixed horizontal scale bars in microscopy image sequences.

The detector is intended for videos where the scale bar is a bright horizontal
line in the lower-right corner and the text label is constant. It detects only
the bar line, writes per-frame coordinates, summarizes length stability, and
optionally saves annotated sample crops for visual QA.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class Detection:
    frame: str
    x: int
    y: int
    width: int
    height: int
    x1: int
    y1: float
    x2: int
    y2: float
    length_px: int
    candidate_count: int
    threshold: int
    kernel_width: int
    kernel_height: int
    roi_x0: int
    roi_y0: int
    roi_x1: int
    roi_y1: int
    frame_index: int | None = None
    scale_value: float | None = None
    scale_unit: str | None = None
    scale_label: str | None = None
    unit_per_pixel: float | None = None


def parse_box(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = [p for p in value.replace(",", " ").split() if p]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Expected four integers: left,top,right,bottom"
        )
    try:
        left, top, right, bottom = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI box values must be integers") from exc
    if right <= left or bottom <= top:
        raise argparse.ArgumentTypeError("ROI right/bottom must exceed left/top")
    return left, top, right, bottom


def list_images(input_dir: Path) -> list[Path]:
    images = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise SystemExit(f"No image files found in {input_dir}")
    return images


def parse_frame_index(frame_name: str) -> int | None:
    matches = re.findall(r"\d+", Path(frame_name).stem)
    if not matches:
        return None
    return int(matches[-1])


def format_scale_value(value: float) -> str:
    return f"{value:g}"


def scale_value_for_frame(
    frame_name: str, args: argparse.Namespace
) -> tuple[int | None, float | None, str | None, str | None]:
    frame_index = parse_frame_index(frame_name)
    if args.scale_switch_frame is None:
        value = args.scale_value
    elif frame_index is None:
        value = None
    elif frame_index < args.scale_switch_frame:
        value = args.scale_value_before
    else:
        value = args.scale_value_after

    if value is None:
        return frame_index, None, args.scale_unit, None
    label = format_scale_value(value)
    if args.scale_unit:
        label = f"{label} {args.scale_unit}"
    return frame_index, float(value), args.scale_unit, label


def attach_scale_info(detection: Detection, args: argparse.Namespace) -> Detection:
    frame_index, scale_value, scale_unit, scale_label = scale_value_for_frame(detection.frame, args)
    unit_per_pixel = None
    if scale_value is not None and detection.length_px > 0:
        unit_per_pixel = float(scale_value) / float(detection.length_px)

    data = asdict(detection)
    data.update(
        {
            "frame_index": frame_index,
            "scale_value": scale_value,
            "scale_unit": scale_unit,
            "scale_label": scale_label,
            "unit_per_pixel": unit_per_pixel,
        }
    )
    return Detection(**data)


def resolve_input_dir(path: Path) -> Path:
    """Accept either an image directory or an ISAT-style root containing png/."""
    path = path.resolve()
    if not path.exists():
        raise SystemExit(f"Input path does not exist: {path}")
    if not path.is_dir():
        raise SystemExit(f"Input path must be a directory: {path}")
    if any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in path.iterdir()):
        return path
    png_dir = path / "png"
    if png_dir.is_dir() and any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in png_dir.iterdir()
    ):
        return png_dir.resolve()
    raise SystemExit(
        f"Input directory has no images and no png/ image subdirectory: {path}"
    )


def default_output_dir(input_dir: Path) -> Path:
    if input_dir.name.lower() == "png":
        return input_dir.parent / "scale_bar_detection"
    return input_dir / "scale_bar_detection"


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"Output directory already exists and is not empty: {path}\n"
                "Use --overwrite or choose a different --output-dir."
            )
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def read_color(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image as {path.suffix}: {path}")
    encoded.tofile(str(path))


def compute_roi(
    width: int,
    height: int,
    roi_box: tuple[int, int, int, int] | None,
    roi_x_frac: float,
    roi_y_frac: float,
) -> tuple[int, int, int, int]:
    if roi_box is None:
        x0 = int(width * roi_x_frac)
        y0 = int(height * roi_y_frac)
        x1 = width
        y1 = height
    else:
        x0, y0, x1, y1 = roi_box
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height or x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"ROI {(x0, y0, x1, y1)} is invalid for image size {width}x{height}"
        )
    return x0, y0, x1, y1


def detect_one(
    gray: np.ndarray,
    frame_name: str,
    threshold: int,
    kernel_size: tuple[int, int],
    roi_box: tuple[int, int, int, int] | None,
    roi_x_frac: float,
    roi_y_frac: float,
    polarity: str,
    min_width: int,
    max_height: int,
    min_ratio: float,
) -> tuple[Detection | None, list[dict[str, float | int | str]], tuple[int, int, int, int]]:
    height, width = gray.shape[:2]
    x0, y0, x1, y1 = compute_roi(width, height, roi_box, roi_x_frac, roi_y_frac)
    roi = gray[y0:y1, x0:x1]

    mode = cv2.THRESH_BINARY if polarity == "bright" else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(roi, threshold, 255, mode)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict[str, float | int | str]] = []
    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        ratio = box_w / max(box_h, 1)
        if box_w >= min_width and box_h <= max_height and ratio >= min_ratio:
            candidates.append(
                {
                    "frame": frame_name,
                    "x": x0 + int(x),
                    "y": y0 + int(y),
                    "width": int(box_w),
                    "height": int(box_h),
                    "aspect_ratio": float(ratio),
                    "area": float(cv2.contourArea(contour)),
                }
            )

    if not candidates:
        return None, candidates, (x0, y0, x1, y1)

    best = max(
        candidates,
        key=lambda item: (
            int(item["width"]),
            int(item["x"]) + int(item["y"]),
            float(item["area"]),
        ),
    )
    x = int(best["x"])
    y = int(best["y"])
    box_w = int(best["width"])
    box_h = int(best["height"])
    y_center = y + (box_h - 1) / 2.0
    detection = Detection(
        frame=frame_name,
        x=x,
        y=y,
        width=box_w,
        height=box_h,
        x1=x,
        y1=y_center,
        x2=x + box_w - 1,
        y2=y_center,
        length_px=box_w,
        candidate_count=len(candidates),
        threshold=threshold,
        kernel_width=kernel_size[0],
        kernel_height=kernel_size[1],
        roi_x0=x0,
        roi_y0=y0,
        roi_x1=x1,
        roi_y1=y1,
    )
    return detection, candidates, (x0, y0, x1, y1)


def write_csv(path: Path, rows: list[Detection]) -> None:
    compatibility_fields = ["image", "pixel_length", "unit"]
    detection_fields = [
        name for name in Detection.__annotations__ if name not in compatibility_fields
    ]
    fieldnames = compatibility_fields + detection_fields
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            writer.writerow(
                {
                    "image": row.frame,
                    "pixel_length": row.length_px,
                    "unit": row.scale_unit,
                    **data,
                }
            )


def summarize(
    input_dir: Path,
    output_dir: Path,
    total_frames: int,
    detections: list[Detection],
    failures: list[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, object]:
    lengths = [d.length_px for d in detections]
    boxes = [(d.x, d.y, d.width, d.height) for d in detections]
    box_counts = Counter(boxes)
    length_counts = Counter(lengths)
    scale_groups: dict[str, dict[str, object]] = {}
    for label in sorted({d.scale_label for d in detections if d.scale_label}):
        group = [d for d in detections if d.scale_label == label]
        group_lengths = [d.length_px for d in group]
        group_units = [d.unit_per_pixel for d in group if d.unit_per_pixel is not None]
        scale_groups[str(label)] = {
            "count": len(group),
            "length_px_min": min(group_lengths) if group_lengths else None,
            "length_px_median": float(np.median(group_lengths)) if group_lengths else None,
            "length_px_max": max(group_lengths) if group_lengths else None,
            "unit_per_pixel_median": float(np.median(group_units)) if group_units else None,
        }

    summary: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "frame_count": total_frames,
        "detected_count": len(detections),
        "failure_count": len(failures),
        "failures": failures,
        "threshold": args.threshold,
        "polarity": args.polarity,
        "kernel_size": [args.kernel_width, args.kernel_height],
        "roi": {
            "box": list(args.roi_box) if args.roi_box else None,
            "x_fraction": args.roi_x_frac,
            "y_fraction": args.roi_y_frac,
        },
        "filters": {
            "min_width": args.min_width,
            "max_height": args.max_height,
            "min_ratio": args.min_ratio,
        },
        "length_px_min": min(lengths) if lengths else None,
        "length_px_median": float(np.median(lengths)) if lengths else None,
        "length_px_max": max(lengths) if lengths else None,
        "length_px_mode": length_counts.most_common(1)[0][0] if lengths else None,
        "length_px_counts": dict(sorted(length_counts.items())) if lengths else {},
        "most_common_bbox": None,
        "scale_value": args.scale_value,
        "scale_unit": args.scale_unit,
        "scale_switch": (
            {
                "switch_frame": args.scale_switch_frame,
                "before_value": args.scale_value_before,
                "after_value": args.scale_value_after,
                "unit": args.scale_unit,
                "rule": "frame < switch_frame uses before_value; frame >= switch_frame uses after_value",
            }
            if args.scale_switch_frame is not None
            else None
        ),
        "unit_per_pixel": None,
        "scale_groups": scale_groups,
    }
    if boxes:
        bbox, count = box_counts.most_common(1)[0]
        summary["most_common_bbox"] = {
            "x": bbox[0],
            "y": bbox[1],
            "width": bbox[2],
            "height": bbox[3],
            "count": count,
        }
    if args.scale_value is not None and lengths:
        summary["unit_per_pixel"] = float(args.scale_value) / float(np.median(lengths))
    return summary


def sample_indices(count: int, sample_count: int) -> list[int]:
    if count <= 0 or sample_count <= 0:
        return []
    sample_count = min(sample_count, count)
    if sample_count == 1:
        return [0]
    return sorted(
        {int(round(i * (count - 1) / (sample_count - 1))) for i in range(sample_count)}
    )


def write_annotated_samples(
    images: list[Path],
    detections: list[Detection],
    output_dir: Path,
    sample_count: int,
    crop_margin_x: int,
    crop_margin_y: int,
) -> list[str]:
    if sample_count <= 0 or not detections:
        return []

    by_frame = {d.frame: d for d in detections}
    detected_images = [p for p in images if p.name in by_frame]
    samples_dir = output_dir / "annotated_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for idx in sample_indices(len(detected_images), sample_count):
        path = detected_images[idx]
        detection = by_frame[path.name]
        image = read_color(path)
        x, y, width, height = detection.x, detection.y, detection.width, detection.height

        cv2.rectangle(
            image,
            (x, y),
            (x + width - 1, y + height - 1),
            (0, 0, 255),
            4,
        )
        cv2.line(
            image,
            (detection.x1, int(round(detection.y1))),
            (detection.x2, int(round(detection.y2))),
            (0, 255, 255),
            2,
        )
        if detection.scale_label and detection.unit_per_pixel is not None:
            label = (
                f"scale bar: {detection.length_px} px = {detection.scale_label}, "
                f"{detection.unit_per_pixel:.5g} {detection.scale_unit or 'unit'}/px"
            )
        else:
            label = f"scale bar: {detection.length_px} px"
        cv2.putText(
            image,
            label,
            (max(0, x - 80), max(35, y - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

        y0 = max(0, y - crop_margin_y)
        y1 = min(image.shape[0], y + height + crop_margin_y)
        x0 = max(0, x - crop_margin_x)
        x1 = min(image.shape[1], x + width + crop_margin_x)
        crop = image[y0:y1, x0:x1]
        out_path = samples_dir / path.name
        write_image(out_path, crop)
        written.append(str(out_path))
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect lower-right horizontal scale bars in image sequences."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Image directory, or an ISAT-style root containing a png/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <input root>/scale_bar_detection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing non-empty output directory.",
    )
    parser.add_argument("--threshold", type=int, default=220)
    parser.add_argument(
        "--polarity",
        choices=["bright", "dark"],
        default="bright",
        help="Use bright for white bars, dark for black bars.",
    )
    parser.add_argument(
        "--kernel-width",
        type=int,
        default=35,
        help="Horizontal morphology width in pixels (default: 35).",
    )
    parser.add_argument("--kernel-height", type=int, default=5)
    parser.add_argument(
        "--roi-x-frac",
        type=float,
        default=0.55,
        help="Default ROI left edge as image-width fraction.",
    )
    parser.add_argument(
        "--roi-y-frac",
        type=float,
        default=0.60,
        help="Default ROI top edge as image-height fraction.",
    )
    parser.add_argument(
        "--roi-box",
        type=parse_box,
        default=None,
        help="Optional explicit ROI: left,top,right,bottom.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=35,
        help="Minimum detected bar width in pixels (default: 35).",
    )
    parser.add_argument("--max-height", type=int, default=35)
    parser.add_argument("--min-ratio", type=float, default=5.0)
    parser.add_argument(
        "--scale-value",
        type=float,
        default=None,
        help="Known scale label value, for example 20.",
    )
    parser.add_argument(
        "--scale-unit",
        default=None,
        help="Known scale label unit, for example nm. Used only in summary.",
    )
    parser.add_argument(
        "--scale-switch-frame",
        type=int,
        default=None,
        help="Frame index where the scale label switches; frames before it use --scale-value-before.",
    )
    parser.add_argument(
        "--scale-value-before",
        type=float,
        default=None,
        help="Scale label value before --scale-switch-frame.",
    )
    parser.add_argument(
        "--scale-value-after",
        type=float,
        default=None,
        help="Scale label value at and after --scale-switch-frame.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="Number of annotated sample crops to save. Use 0 to disable.",
    )
    parser.add_argument("--sample-margin-x", type=int, default=280)
    parser.add_argument("--sample-margin-y", type=int, default=180)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = resolve_input_dir(args.input_dir)
    output_dir = (args.output_dir or default_output_dir(input_dir)).resolve()
    prepare_output_dir(output_dir, args.overwrite)

    images = list_images(input_dir)
    detections: list[Detection] = []
    failures: list[dict[str, str]] = []
    all_candidates: list[dict[str, float | int | str]] = []

    kernel_size = (args.kernel_width, args.kernel_height)
    if args.kernel_width <= 0 or args.kernel_height <= 0:
        raise SystemExit("Kernel width/height must be positive.")
    if not 0 <= args.threshold <= 255:
        raise SystemExit("--threshold must be in 0..255.")
    if args.scale_switch_frame is not None and (
        args.scale_value_before is None or args.scale_value_after is None
    ):
        raise SystemExit(
            "--scale-switch-frame requires --scale-value-before and --scale-value-after."
        )

    for path in images:
        try:
            gray = read_gray(path)
            detection, candidates, _roi = detect_one(
                gray=gray,
                frame_name=path.name,
                threshold=args.threshold,
                kernel_size=kernel_size,
                roi_box=args.roi_box,
                roi_x_frac=args.roi_x_frac,
                roi_y_frac=args.roi_y_frac,
                polarity=args.polarity,
                min_width=args.min_width,
                max_height=args.max_height,
                min_ratio=args.min_ratio,
            )
        except Exception as exc:
            failures.append({"frame": path.name, "reason": str(exc)})
            continue
        all_candidates.extend(candidates)
        if detection is None:
            failures.append({"frame": path.name, "reason": "no candidate"})
            continue
        detections.append(attach_scale_info(detection, args))

    detections_csv = output_dir / "scale_bar_detections.csv"
    detections_json = output_dir / "scale_bar_detections.json"
    candidates_json = output_dir / "scale_bar_candidates.json"
    summary_json = output_dir / "scale_bar_summary.json"

    write_csv(detections_csv, detections)
    with detections_json.open("w", encoding="utf-8") as handle:
        json.dump([asdict(row) for row in detections], handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with candidates_json.open("w", encoding="utf-8") as handle:
        json.dump(all_candidates, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    sample_paths = write_annotated_samples(
        images=images,
        detections=detections,
        output_dir=output_dir,
        sample_count=args.sample_count,
        crop_margin_x=args.sample_margin_x,
        crop_margin_y=args.sample_margin_y,
    )

    summary = summarize(input_dir, output_dir, len(images), detections, failures, args)
    summary["outputs"] = {
        "detections_csv": str(detections_csv),
        "detections_json": str(detections_json),
        "candidates_json": str(candidates_json),
        "summary_json": str(summary_json),
        "annotated_samples": sample_paths,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
