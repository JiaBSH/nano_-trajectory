"""Rendering overlays on annotation-sized images."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class AnnotatedImageMixin:
    """Provide rendering overlays on annotation-sized images."""

    def annotate_images(
        self,
        output_dir=None,
        label_ids=False,
        id_mode="event",
        max_dist=50.0,
        min_track_length=0,
        use_display_id=True,
    ):
        output_dir = self._resolve_annotation_output_dir(output_dir, "annotated_images")

        if not self.image_path or not self.image_dir:
            print("[skip] annotate_images: image_path is not set.")
            return

        print(f"Annotating images to {output_dir}...")

        try:
            # Try to start with a slightly larger font if possible
            font = ImageFont.truetype("arial.ttf", 38)
        except OSError:
            font = ImageFont.load_default()

        assigned_ids_by_frame = None
        display_id_of = None
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
                        "annotate_images(label_ids=True) currently supports id_mode='event' only"
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
                if bool(use_display_id):
                    display_id_of = self._display_id_mapping(series_by_id_for_display)
                else:
                    display_id_of = None

                print(
                    f"Annotate IDs enabled: mode={mode}, max_dist_nm={float(max_dist)}, "
                    f"min_track_length={int(min_track_length)}, use_display_id={bool(use_display_id)}, "
                    f"ids_total={len(series_by_id)}"
                )

        # For robust annotation across categories:
        # - Draw the segmentation contours for self.target_category.
        # - Only for nanodroplet, additionally draw diameter/height overlays.
        for frame_id, json_name in enumerate(self.json_files):
            frame_name = Path(json_name).stem
            # Find image
            img_path = None
            possible_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
            for ext in possible_exts:
                p = os.path.join(self.image_dir, frame_name + ext)
                if os.path.exists(p):
                    img_path = p
                    break
            if not img_path:
                continue

            try:
                with Image.open(img_path) as im:
                    img_out = im.convert("RGB")
                    draw = ImageDraw.Draw(img_out)

                    json_path = os.path.join(self.json_dir, json_name)
                    with open(json_path, "r", encoding="utf-8") as f:
                        jdata = json.load(f)

                    try:
                        nm_per_px = self._nm_per_px_for_frame(frame_name)
                    except Exception:
                        nm_per_px = None

                    ids_this_frame = None
                    if assigned_ids_by_frame is not None:
                        ids_this_frame = assigned_ids_by_frame.get(int(frame_id))

                    obj_idx = 0

                    for obj in jdata.get("objects", []):
                        if obj.get("category") != self.target_category:
                            continue
                        pts_raw = np.array(
                            obj.get("segmentation", []), dtype=np.float32
                        )
                        if pts_raw.shape[0] < 3:
                            obj_idx += 1
                            continue

                        # ID label (use the same within-frame order as JSON/category iteration)
                        if (
                            bool(label_ids)
                            and ids_this_frame is not None
                            and obj_idx < len(ids_this_frame)
                        ):
                            try:
                                instance_id = int(ids_this_frame[obj_idx])
                                if display_id_of is not None:
                                    disp = int(display_id_of.get(instance_id, 0))
                                    id_text = (
                                        str(disp) if disp > 0 else str(instance_id)
                                    )
                                else:
                                    id_text = str(instance_id)

                                cx_px = float(np.mean(pts_raw[:, 0]))
                                cy_px = float(np.mean(pts_raw[:, 1]))
                                r = 6
                                draw.ellipse(
                                    (cx_px - r, cy_px - r, cx_px + r, cy_px + r),
                                    outline="orange",
                                    width=3,
                                )
                                draw.text(
                                    (cx_px - 20, cy_px - 18),
                                    id_text,
                                    fill="orange",
                                    font=font,
                                    stroke_width=2,
                                    stroke_fill="black",
                                )
                            except Exception:
                                pass

                        # draw contour
                        poly = [tuple(map(float, p)) for p in pts_raw]
                        draw.polygon(poly, outline="lime", width=2)

                        # droplet-only: draw diameter/height overlay
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
                                # best-effort; keep contour even if dims fail
                                pass

                        obj_idx += 1

                    out_path = os.path.join(output_dir, frame_name + ".png")
                    img_out.save(out_path)

            except Exception as e:
                print(f"Error annotating {frame_name}: {e}")
