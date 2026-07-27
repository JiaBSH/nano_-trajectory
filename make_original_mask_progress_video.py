from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


DEFAULT_IMAGE_DIR = Path(r"D:\code\zwl_NANO\data\zwl1\merged_forward_original_size\png")
DEFAULT_LABEL_DIR = Path(r"D:\code\zwl_NANO\data\zwl1\merged_forward_original_size\label")
DEFAULT_OUTPUT = Path(
    r"D:\code\zwl_NANO\data\zwl1\merged_forward_original_size\frame_visualization"
    r"\original_vs_mask_progress.mp4"
)

# Same category colors used by analyze-rawframe.py.
CATEGORY_COLORS_RGB = {
    "nanocluster": (220, 30, 30),
    "nanodroplet": (30, 100, 255),
    "gas": (0, 200, 80),
}
DEFAULT_CATEGORY_RGB = (255, 140, 0)
SKIP_CATEGORIES = {"pin"}

VALID_BAR_RGB = (255, 210, 0)
MISSING_BAR_RGB = (0, 0, 0)
POINTER_RGB = (255, 255, 255)
POINTER_OUTLINE_RGB = (0, 0, 0)


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def parse_frame_index(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def collect_indexed_files(folder: Path, suffix: str) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() != suffix.lower():
            continue
        idx = parse_frame_index(path)
        if idx is not None and idx not in files:
            files[idx] = path
    return files


def first_valid_image(image_files: dict[int, Path], label_files: dict[int, Path]) -> np.ndarray:
    for idx in sorted(set(image_files) & set(label_files)):
        image = cv2.imread(str(image_files[idx]), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    raise RuntimeError("No readable frame with both image and label was found.")


def normalize_segmentation(segmentation: object) -> np.ndarray | None:
    try:
        points = np.asarray(segmentation, dtype=np.float32)
    except Exception:
        return None

    if points.ndim == 1:
        if points.size < 6 or points.size % 2 != 0:
            return None
        points = points.reshape((-1, 2))

    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return None

    return np.rint(points[:, :2]).astype(np.int32).reshape((-1, 1, 2))


def overlay_masks(image_bgr: np.ndarray, label_path: Path, mask_alpha: int, outline_width: int) -> np.ndarray:
    with label_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    alpha = max(0.0, min(1.0, float(mask_alpha) / 255.0))
    overlay = image_bgr.copy()
    polygons: list[tuple[np.ndarray, tuple[int, int, int]]] = []

    for obj in data.get("objects", []):
        category = str(obj.get("category", "")).strip().lower()
        if category in SKIP_CATEGORIES:
            continue
        points = normalize_segmentation(obj.get("segmentation", []))
        if points is None:
            continue
        color_bgr = rgb_to_bgr(CATEGORY_COLORS_RGB.get(category, DEFAULT_CATEGORY_RGB))
        cv2.fillPoly(overlay, [points], color_bgr, lineType=cv2.LINE_AA)
        polygons.append((points, color_bgr))

    if polygons:
        out = cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0.0)
    else:
        out = image_bgr.copy()

    for points, color_bgr in polygons:
        cv2.polylines(out, [points], isClosed=True, color=color_bgr, thickness=outline_width, lineType=cv2.LINE_AA)

    return out


def resize_to(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    if image_bgr.shape[1] == width and image_bgr.shape[0] == height:
        return image_bgr
    interpolation = cv2.INTER_AREA if image_bgr.shape[1] > width or image_bgr.shape[0] > height else cv2.INTER_LINEAR
    return cv2.resize(image_bgr, (width, height), interpolation=interpolation)


def make_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def build_progress_base(valid_flags: list[bool], width: int, height: int) -> np.ndarray:
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    total = len(valid_flags)
    yellow = rgb_to_bgr(VALID_BAR_RGB)
    black = rgb_to_bgr(MISSING_BAR_RGB)

    for pos, is_valid in enumerate(valid_flags):
        x0 = (pos * width) // total
        x1 = ((pos + 1) * width) // total
        if x1 <= x0:
            x1 = min(width, x0 + 1)
        bar[:, x0:x1] = yellow if is_valid else black

    return bar


def draw_progress_pointer(bar: np.ndarray, pos: int, total: int) -> np.ndarray:
    out = bar.copy()
    height, width = out.shape[:2]
    x = int(round((pos + 0.5) * width / max(total, 1)))
    x = max(0, min(width - 1, x))

    outline = rgb_to_bgr(POINTER_OUTLINE_RGB)
    pointer = rgb_to_bgr(POINTER_RGB)
    line_outer = max(5, width // 900)
    line_inner = max(2, line_outer // 2)

    cv2.line(out, (x, 0), (x, height - 1), outline, thickness=line_outer, lineType=cv2.LINE_AA)
    cv2.line(out, (x, 0), (x, height - 1), pointer, thickness=line_inner, lineType=cv2.LINE_AA)

    tri_h = max(12, height // 3)
    tri_w = max(10, tri_h // 2)
    triangle = np.array(
        [[x, tri_h], [max(0, x - tri_w), 0], [min(width - 1, x + tri_w), 0]],
        dtype=np.int32,
    )
    cv2.polylines(out, [triangle], isClosed=True, color=outline, thickness=max(2, line_inner), lineType=cv2.LINE_AA)
    cv2.fillConvexPoly(out, triangle, pointer, lineType=cv2.LINE_AA)

    return out


def make_writer(output_path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    return writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a side-by-side raw/segmentation video with yellow/black progress bar."
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=float, default=1.0, help="Output scale applied to each original frame.")
    parser.add_argument("--bar-height", type=int, default=96)
    parser.add_argument("--mask-alpha", type=int, default=120)
    parser.add_argument("--outline-width", type=int, default=2)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scale <= 0:
        raise ValueError("--scale must be positive.")

    image_files = collect_indexed_files(args.image_dir, ".png")
    label_files = collect_indexed_files(args.label_dir, ".json")
    if not image_files:
        raise RuntimeError(f"No PNG files found in {args.image_dir}")
    if not label_files:
        raise RuntimeError(f"No JSON label files found in {args.label_dir}")

    start = args.start if args.start is not None else min(min(image_files), min(label_files))
    end = args.end if args.end is not None else max(max(image_files), max(label_files))
    if end < start:
        raise ValueError("--end must be greater than or equal to --start.")

    frame_indices = list(range(start, end + 1))
    valid_flags = [(idx in image_files and idx in label_files) for idx in frame_indices]

    sample = first_valid_image(image_files, label_files)
    source_h, source_w = sample.shape[:2]
    frame_w = make_even(max(2, int(round(source_w * args.scale))))
    frame_h = make_even(max(2, int(round(source_h * args.scale))))
    bar_h = make_even(max(2, int(args.bar_height)))
    video_w = frame_w * 2
    video_h = frame_h + bar_h

    writer = make_writer(args.output, args.fps, (video_w, video_h))
    progress_base = build_progress_base(valid_flags, video_w, bar_h)
    blank_top = np.zeros((frame_h, video_w, 3), dtype=np.uint8)

    missing_count = 0
    failed_count = 0
    total = len(frame_indices)

    try:
        from tqdm import tqdm

        iterator = tqdm(enumerate(frame_indices), total=total, desc="video", unit="frame")
    except Exception:
        iterator = enumerate(frame_indices)

    try:
        for pos, idx in iterator:
            top = blank_top.copy()

            if valid_flags[pos]:
                image = cv2.imread(str(image_files[idx]), cv2.IMREAD_COLOR)
                if image is None:
                    failed_count += 1
                else:
                    try:
                        overlay = overlay_masks(image, label_files[idx], args.mask_alpha, args.outline_width)
                        image = resize_to(image, frame_w, frame_h)
                        overlay = resize_to(overlay, frame_w, frame_h)
                        top[:, :frame_w] = image
                        top[:, frame_w:] = overlay
                    except Exception as exc:
                        failed_count += 1
                        print(f"[warn] frame {idx} failed and was rendered blank: {exc}", flush=True)
            else:
                missing_count += 1

            bar = draw_progress_pointer(progress_base, pos, total)
            canvas = np.vstack((top, bar))
            writer.write(canvas)

            if pos == 0 or (pos + 1) % 50 == 0 or pos + 1 == total:
                print(f"[progress] {pos + 1}/{total} frames", flush=True)
    finally:
        writer.release()

    print(f"[done] output={args.output}", flush=True)
    print(
        f"[summary] range={start}-{end}, frames={total}, valid={sum(valid_flags)}, "
        f"missing_or_unpaired={missing_count}, failed={failed_count}, size={video_w}x{video_h}, fps={args.fps}",
        flush=True,
    )


if __name__ == "__main__":
    main()
