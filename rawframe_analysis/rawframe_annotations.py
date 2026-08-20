"""Rendering gas overlays on original raw frames."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class RawFrameAnnotationMixin:
    """Provide rendering gas overlays on original raw frames."""

    def annotate_images_on_rawframe(
        self,
        raw_frame_dir,
        output_dir=None,
        label_ids=False,
        id_mode="event",
        max_dist=50.0,
        min_track_length=3,
        use_display_id=True,
        mask_alpha=120,
        frame_step=1,
    ):
        """Generate annotated images using the original raw frames as background.

        Semi-transparent filled masks are drawn for each detected object, plus
        contour outlines and analysis annotations (IDs, diameter/height).

        Args:
            raw_frame_dir: Directory containing the original raw frame images.
                           Frame filenames must match JSON stem names.
            output_dir: Where to save results. Defaults to
                        <output_root>/annotated_rawframe. Relative paths are
                        joined with output_root.
            label_ids: Whether to draw instance ID labels.
            id_mode: Tracking mode – only 'event' is supported.
            max_dist: Maximum linking distance in nm for ID assignment.
            min_track_length: Minimum number of frames a track must span to
                              receive a display ID label.
            use_display_id: Remap internal IDs to compact 1-based display IDs.
            mask_alpha: Alpha value (0-255) for filled mask overlays.
            frame_step: Save one annotated image every N frames. Use 1 for all frames.
        """
        frame_step = self._normalize_frame_step(frame_step)

        output_dir = self._resolve_annotation_output_dir(
            output_dir, "annotated_rawframe"
        )

        print(
            f"Annotating raw-frame images to {output_dir} (frame_step={frame_step})..."
        )

        # Font is initialised per-frame based on image size to avoid oversized labels.

        # Category-based fixed colour: nanocluster=red, nanodroplet=blue, gas=green, others=orange
        _CATEGORY_COLOR = {
            "nanocluster": (220, 30, 30),
            "nanodroplet": (30, 100, 255),
            "gas": (0, 200, 80),
        }
        _cat_rgb = _CATEGORY_COLOR.get(str(self.target_category).lower(), (255, 140, 0))
        _fill_rgba = _cat_rgb + (mask_alpha,)
        _outline_rgb = _cat_rgb

        # ---- Build ID assignments (same logic as annotate_images) ----
        assigned_ids_by_frame = None
        display_id_of = None
        allowed_instance_ids = None
        if bool(label_ids):
            if len(self.object_records) == 0:
                print(
                    "[warn] label_ids=True but object_records is empty; run process_all_frames() first."
                )
            else:
                from collections import defaultdict

                detections_by_frame = defaultdict(list)
                for (
                    frame_id,
                    frame_name,
                    nm_per_px,
                    cx_nm,
                    cy_nm,
                    area_nm2,
                ) in self.object_records:
                    detections_by_frame[int(frame_id)].append(
                        (
                            str(frame_name),
                            float(nm_per_px),
                            float(cx_nm),
                            float(cy_nm),
                            float(area_nm2),
                        )
                    )

                mode = str(id_mode).strip().lower()
                if mode != "event":
                    raise NotImplementedError(
                        "annotate_images_on_rawframe(label_ids=True) supports id_mode='event' only"
                    )

                series_by_id, assigned_ids_by_frame, _events = (
                    self._build_event_id_series_with_assignments(
                        detections_by_frame, max_dist=max_dist
                    )
                )

                series_by_id_for_display = {
                    k: v
                    for k, v in series_by_id.items()
                    if len(v) >= int(min_track_length)
                }
                allowed_instance_ids = set(
                    int(k) for k in series_by_id_for_display.keys()
                )
                if bool(use_display_id):
                    display_id_of = self._display_id_mapping(series_by_id_for_display)
                else:
                    display_id_of = None

                print(
                    f"Annotate IDs enabled: mode={mode}, max_dist_nm={float(max_dist)}, "
                    f"min_track_length={int(min_track_length)}, use_display_id={bool(use_display_id)}, "
                    f"ids_total={len(series_by_id)}"
                )

        # ---- Per-frame annotation ----
        possible_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        saved_count = 0

        for frame_id, json_name in enumerate(self.json_files):
            if int(frame_id) % frame_step != 0:
                continue

            frame_name = Path(json_name).stem

            # Locate raw frame image
            raw_img_path = None
            for ext in possible_exts:
                p = os.path.join(raw_frame_dir, frame_name + ext)
                if os.path.exists(p):
                    raw_img_path = p
                    break
            if not raw_img_path:
                continue

            try:
                with Image.open(raw_img_path) as raw_im:
                    # Convert to RGBA so we can composite a mask layer
                    bg = raw_im.convert("RGBA")
                    W, H = bg.size

                    # Adaptive text size for different frame resolutions.
                    # Example: 512px frame -> ~14px font; 1024px frame -> ~24px font.
                    font_px = max(18, min(30, int(round(min(W, H) * 0.028))))
                    try:
                        font = ImageFont.truetype("arial.ttf", font_px)
                    except OSError:
                        font = ImageFont.load_default()
                    id_offset_x = int(max(8, round(font_px * 0.8)))
                    id_offset_y = int(max(8, round(font_px * 0.7)))
                    centroid_r = int(max(3, round(font_px * 0.22)))
                    stroke_w = int(max(1, round(font_px * 0.12)))

                    # Transparent overlay for filled masks
                    mask_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                    mask_draw = ImageDraw.Draw(mask_layer)

                    json_path = os.path.join(self.json_dir, json_name)
                    with open(json_path, "r", encoding="utf-8") as f:
                        jdata = json.load(f)
                    jdata = self._postprocess_frame_instances(
                        jdata, frame_name=frame_name
                    )

                    try:
                        nm_per_px = self._nm_per_px_for_frame(frame_name)
                    except Exception:
                        nm_per_px = None

                    ids_this_frame = None
                    if assigned_ids_by_frame is not None:
                        ids_this_frame = assigned_ids_by_frame.get(int(frame_id))

                    obj_idx = 0
                    objects_this_frame = [
                        obj
                        for obj in jdata.get("objects", [])
                        if obj.get("category") == self.target_category
                    ]

                    for obj in objects_this_frame:
                        pts_raw = np.array(
                            obj.get("segmentation", []), dtype=np.float32
                        )
                        if pts_raw.shape[0] < 3:
                            obj_idx += 1
                            continue

                        # Draw filled semi-transparent mask (uniform colour per category)
                        poly = [tuple(map(float, p)) for p in pts_raw]
                        mask_draw.polygon(poly, fill=_fill_rgba, outline=None)

                        obj_idx += 1

                    # Composite mask layer onto background
                    img_out = Image.alpha_composite(bg, mask_layer).convert("RGB")
                    draw = ImageDraw.Draw(img_out)

                    # Second pass: outlines + text on top of composited image
                    obj_idx = 0
                    for obj in objects_this_frame:
                        pts_raw = np.array(
                            obj.get("segmentation", []), dtype=np.float32
                        )
                        if pts_raw.shape[0] < 3:
                            obj_idx += 1
                            continue

                        # Draw contour outline
                        poly = [tuple(map(float, p)) for p in pts_raw]
                        draw.polygon(poly, outline=_outline_rgb, width=2)

                        # ID label
                        if (
                            bool(label_ids)
                            and ids_this_frame is not None
                            and obj_idx < len(ids_this_frame)
                        ):
                            try:
                                instance_id = int(ids_this_frame[obj_idx])
                                draw_id_label = True

                                # Hide short-lived/noisy tracks to reduce excessive labels.
                                if (
                                    allowed_instance_ids is not None
                                    and instance_id not in allowed_instance_ids
                                ):
                                    draw_id_label = False

                                if display_id_of is not None:
                                    disp = int(display_id_of.get(instance_id, 0))
                                    if disp <= 0:
                                        draw_id_label = False
                                    id_text = str(disp) if disp > 0 else ""
                                else:
                                    id_text = str(instance_id)

                                if draw_id_label:
                                    cx_px = float(np.mean(pts_raw[:, 0]))
                                    cy_px = float(np.mean(pts_raw[:, 1]))
                                    r = centroid_r
                                    draw.ellipse(
                                        (cx_px - r, cy_px - r, cx_px + r, cy_px + r),
                                        outline=_outline_rgb,
                                        width=2,
                                    )
                                    draw.text(
                                        (cx_px - id_offset_x, cy_px - id_offset_y),
                                        id_text,
                                        fill="orange",
                                        font=font,
                                        stroke_width=stroke_w,
                                        stroke_fill="black",
                                    )
                            except Exception:
                                pass

                        # Diameter / height overlay (nanodroplet only)
                        if (
                            str(self.target_category).lower() == "nanodroplet"
                            and nm_per_px is not None
                        ):
                            try:
                                d_px, h_px, box_info = (
                                    self._compute_droplet_dims_oriented(pts_raw)
                                )
                                d_nm = float(d_px) * float(nm_per_px)
                                h_nm = float(h_px) * float(nm_per_px)

                                corners = box_info.get("corners")
                                if corners is not None:
                                    corners_arr = np.array(corners, dtype=np.float32)
                                    rect_poly = [
                                        tuple(map(float, p)) for p in corners_arr
                                    ]
                                    draw.polygon(rect_poly, outline="cyan", width=2)
                                    text = f"D:{d_nm:.1f}\nH:{h_nm:.1f}"
                                    cx = float(corners_arr[:, 0].mean())
                                    cy = float(corners_arr[:, 1].mean())
                                    draw.text((cx, cy), text, fill="yellow", font=font)

                                baseline_p1 = box_info.get("baseline_p1")
                                baseline_p2 = box_info.get("baseline_p2")
                                if baseline_p1 is not None and baseline_p2 is not None:
                                    draw.line(
                                        [
                                            tuple(map(float, baseline_p1)),
                                            tuple(map(float, baseline_p2)),
                                        ],
                                        fill="red",
                                        width=3,
                                    )

                                apex = box_info.get("apex_point")
                                base_mid = box_info.get("base_mid_point")
                                if apex is not None and base_mid is not None:
                                    draw.line(
                                        [
                                            tuple(map(float, apex)),
                                            tuple(map(float, base_mid)),
                                        ],
                                        fill="magenta",
                                        width=2,
                                    )
                            except Exception:
                                pass

                        obj_idx += 1

                    out_path = os.path.join(output_dir, frame_name + ".png")
                    img_out.save(out_path)
                    saved_count += 1

            except Exception as e:
                print(f"Error annotating raw frame {frame_name}: {e}")

        print(
            f"Raw-frame annotation complete: {output_dir} ({saved_count} images saved)"
        )
