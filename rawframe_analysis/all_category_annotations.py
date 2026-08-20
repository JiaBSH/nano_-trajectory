"""Rendering every segmentation category on original raw frames."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class AllCategoryAnnotationMixin:
    """Provide rendering every segmentation category on original raw frames."""

    def annotate_boundary_pair_ids_on_rawframe(
        self,
        raw_frame_dir,
        output_dir=None,
        mask_alpha=120,
        show_centroid=False,
        max_dist=50.0,
        frame_step=1,
    ):
        """Render a separate all-category overlay labeled with plot pair IDs.

        Particle labels use ``P1``, ``P2``, ... and droplet labels use ``D1``,
        ``D2``, ... using exactly the same tracking assignments as the boundary
        distance trajectory plots.
        """
        return self.annotate_allcategories_on_rawframe(
            raw_frame_dir=raw_frame_dir,
            output_dir=output_dir or "annotated_boundary_pair_ids_rawframe",
            mask_alpha=mask_alpha,
            show_centroid=show_centroid,
            label_ids=True,
            max_dist=max_dist,
            use_display_id=True,
            frame_step=frame_step,
            boundary_pair_labels=True,
        )

    def _boundary_pair_annotation_ids(self, max_dist):
        """Build plot-aligned P/D display IDs for raw-frame annotation."""
        assignments = {}
        prefixes = {}
        for category, records, prefix in (
            (
                str(self.particle_category).strip().lower(),
                self.boundary_particle_records,
                "P",
            ),
            (
                str(self.droplet_category).strip().lower(),
                self.boundary_droplet_records,
                "D",
            ),
        ):
            assignments[category] = self._tracked_category_ids(
                records, max_dist=float(max_dist), category=category
            )
            prefixes[category] = prefix
        return assignments, prefixes

    def annotate_allcategories_on_rawframe(
        self,
        raw_frame_dir,
        output_dir=None,
        mask_alpha=120,
        show_centroid=True,
        label_ids=True,
        max_dist=50.0,
        use_display_id=True,
        frame_step=1,
        boundary_pair_labels=False,
    ):
        """Generate annotated images using the original raw frames as background,
        drawing masks and outlines for ALL annotation categories in each JSON.

        Category colour mapping:
            nanocluster -> red  (220, 30, 30)
            nanodroplet -> blue (30, 100, 255)
            gas         -> green (0, 200, 80)
            pin         -> yellow (240, 200, 0)  (skipped)
            others      -> orange (255, 140, 0)

        Instance IDs are tracked per category and match the exported CSV tables
        for self.target_category.  Other categories receive their own consistent IDs.

        Args:
            raw_frame_dir: Directory containing the original raw frame images.
            output_dir: Output directory. Defaults to
                        <output_root>/annotated_allcat_rawframe.
            mask_alpha: Alpha value (0-255) for filled mask overlays.
            show_centroid: Whether to draw a centroid dot on each instance.
            label_ids: Whether to draw instance ID labels.
            max_dist: Maximum centroid linking distance (nm) for ID tracking.
            use_display_id: Remap internal IDs to compact 1-based display IDs.
            frame_step: Save one annotated image every N frames. Use 1 for all frames.
            boundary_pair_labels: Internal mode used by
                annotate_boundary_pair_ids_on_rawframe to label only the
                distance-plot categories with matching P/D IDs.
        """
        frame_step = self._normalize_frame_step(frame_step)

        output_dir = self._resolve_annotation_output_dir(
            output_dir, "annotated_allcat_rawframe"
        )

        print(
            f"Annotating all-categories raw-frame images to {output_dir} (frame_step={frame_step})..."
        )

        # Font is initialised per-frame based on image size to avoid oversized labels.

        _CATEGORY_COLOR = {
            "nanocluster": (220, 30, 30),
            "nanodroplet": (30, 100, 255),
            "gas": (0, 200, 80),
            "pin": (240, 200, 0),
        }
        _DEFAULT_COLOR = (255, 140, 0)

        # ---- Pre-scan: build per-category detections for ID tracking ----
        from collections import defaultdict

        # cat_dets[cat][frame_id] = [(frame_name, nm_per_px, cx_nm, cy_nm, area_nm2), ...]
        cat_dets = defaultdict(lambda: defaultdict(list))

        # --- target_category: use self.object_records (drift-corrected, same as CSV export) ---
        _target_cat = str(self.target_category).strip().lower()
        if self.object_records:
            for (
                frame_id_r,
                frame_name_r,
                nm_per_px_r,
                cx_nm_r,
                cy_nm_r,
                area_nm2_r,
            ) in self.object_records:
                cat_dets[_target_cat][int(frame_id_r)].append(
                    (
                        str(frame_name_r),
                        float(nm_per_px_r),
                        float(cx_nm_r),
                        float(cy_nm_r),
                        float(area_nm2_r),
                    )
                )

        # --- other categories: read from JSON (no drift correction available) ---
        for frame_id, json_name in enumerate(self.json_files):
            frame_name = Path(json_name).stem
            json_path = os.path.join(self.json_dir, json_name)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    jdata_pre = json.load(f)
            except Exception:
                continue
            jdata_pre = self._postprocess_frame_instances(
                jdata_pre, frame_name=frame_name
            )
            try:
                nm_pre = self._nm_per_px_for_frame(frame_name)
            except Exception:
                nm_pre = None
            nm_pre_f = float(nm_pre) if nm_pre is not None else 1.0

            for obj in jdata_pre.get("objects", []):
                cat = str(obj.get("category", "")).strip().lower()
                if cat == "pin" or cat == _target_cat:
                    continue  # target_category already handled above
                pts_pre = np.array(obj.get("segmentation", []), dtype=np.float32)
                if pts_pre.shape[0] < 3:
                    continue
                cx_nm = float(np.mean(pts_pre[:, 0])) * nm_pre_f
                cy_nm = float(np.mean(pts_pre[:, 1])) * nm_pre_f
                area_nm2 = self.polygon_area(pts_pre) * nm_pre_f * nm_pre_f
                cat_dets[cat][int(frame_id)].append(
                    (frame_name, nm_pre_f, cx_nm, cy_nm, area_nm2)
                )

        # Build per-category ID assignments
        cat_assigned_ids = {}  # cat -> {frame_id: [id, ...]}
        cat_display_id_of = {}  # cat -> {instance_id: display_id}
        boundary_prefix_by_cat = {}

        if bool(label_ids):
            if bool(boundary_pair_labels):
                cat_assigned_ids, boundary_prefix_by_cat = (
                    self._boundary_pair_annotation_ids(max_dist=max_dist)
                )
                cat_display_id_of = {cat: None for cat in cat_assigned_ids}
            else:
                for cat, det_by_frame in cat_dets.items():
                    series_by_id, assigned_ids, _events = (
                        self._build_event_id_series_with_assignments(
                            det_by_frame, max_dist=float(max_dist)
                        )
                    )
                    cat_assigned_ids[cat] = assigned_ids
                    if bool(use_display_id):
                        cat_display_id_of[cat] = self._display_id_mapping(series_by_id)
                    else:
                        cat_display_id_of[cat] = None

        # ---- Per-frame annotation ----
        possible_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        saved_count = 0

        for frame_id, json_name in enumerate(self.json_files):
            if int(frame_id) % frame_step != 0:
                continue

            frame_name = Path(json_name).stem

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
                    bg = raw_im.convert("RGBA")
                    W, H = bg.size

                    font_px = max(18, min(32, int(round(min(W, H) * 0.029))))
                    try:
                        font = ImageFont.truetype("arial.ttf", font_px)
                    except OSError:
                        font = ImageFont.load_default()
                    id_offset_x = int(max(8, round(font_px * 0.8)))
                    id_offset_y = int(max(8, round(font_px * 0.7)))
                    centroid_r = int(max(3, round(font_px * 0.22)))
                    stroke_w = int(max(1, round(font_px * 0.12)))

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

                    all_objects = jdata.get("objects", [])

                    # First pass: draw filled masks
                    for obj in all_objects:
                        cat = str(obj.get("category", "")).strip().lower()
                        if cat == "pin":
                            continue
                        pts_raw = np.array(
                            obj.get("segmentation", []), dtype=np.float32
                        )
                        if pts_raw.shape[0] < 3:
                            continue
                        rgb = _CATEGORY_COLOR.get(cat, _DEFAULT_COLOR)
                        fill_rgba = rgb + (mask_alpha,)
                        poly = [tuple(map(float, pt)) for pt in pts_raw]
                        mask_draw.polygon(poly, fill=fill_rgba, outline=None)

                    # Composite mask onto background
                    img_out = Image.alpha_composite(bg, mask_layer).convert("RGB")
                    draw = ImageDraw.Draw(img_out)

                    # Second pass: outlines + IDs + centroid + nanodroplet dims
                    # Track per-category object index to match pre-scan ordering
                    cat_obj_idx = defaultdict(int)

                    for obj in all_objects:
                        cat = str(obj.get("category", "")).strip().lower()
                        if cat == "pin":
                            continue
                        pts_raw = np.array(
                            obj.get("segmentation", []), dtype=np.float32
                        )
                        if pts_raw.shape[0] < 3:
                            cat_obj_idx[cat] += 1
                            continue

                        rgb = _CATEGORY_COLOR.get(cat, _DEFAULT_COLOR)
                        poly = [tuple(map(float, pt)) for pt in pts_raw]
                        draw.polygon(poly, outline=rgb, width=2)

                        cx_px = float(np.mean(pts_raw[:, 0]))
                        cy_px = float(np.mean(pts_raw[:, 1]))

                        # Centroid dot
                        if bool(show_centroid):
                            r = centroid_r
                            draw.ellipse(
                                (cx_px - r, cy_px - r, cx_px + r, cy_px + r),
                                fill=rgb,
                                outline="white",
                                width=1,
                            )

                        # Instance ID label
                        if bool(label_ids) and cat in cat_assigned_ids:
                            obj_idx_in_cat = cat_obj_idx[cat]
                            ids_this_frame = cat_assigned_ids[cat].get(int(frame_id))
                            if ids_this_frame is not None and obj_idx_in_cat < len(
                                ids_this_frame
                            ):
                                instance_id = int(ids_this_frame[obj_idx_in_cat])
                                display_map = cat_display_id_of.get(cat)
                                if display_map is not None:
                                    disp = int(display_map.get(instance_id, 0))
                                    id_text = (
                                        str(disp) if disp > 0 else str(instance_id)
                                    )
                                else:
                                    id_text = str(instance_id)
                                prefix = boundary_prefix_by_cat.get(cat, "")
                                if prefix:
                                    id_text = f"{prefix}{id_text}"
                                try:
                                    text_position = (
                                        cx_px - id_offset_x,
                                        cy_px - id_offset_y,
                                    )
                                    if bool(boundary_pair_labels):
                                        text_box = draw.textbbox(
                                            text_position,
                                            id_text,
                                            font=font,
                                            stroke_width=stroke_w,
                                        )
                                        pad = max(2, int(round(font_px * 0.12)))
                                        draw.rounded_rectangle(
                                            (
                                                text_box[0] - pad,
                                                text_box[1] - pad,
                                                text_box[2] + pad,
                                                text_box[3] + pad,
                                            ),
                                            radius=pad,
                                            fill="white",
                                            outline=rgb,
                                            width=max(1, stroke_w),
                                        )
                                    draw.text(
                                        text_position,
                                        id_text,
                                        fill=rgb,
                                        font=font,
                                        stroke_width=stroke_w,
                                        stroke_fill="black",
                                    )
                                except Exception:
                                    pass

                        cat_obj_idx[cat] += 1

                        # Baseline + height overlay: only for nanodroplet objects
                        # AND only when the tracker's focus category is nanodroplet
                        if (
                            not bool(boundary_pair_labels)
                            and cat == "nanodroplet"
                            and str(self.target_category).lower() == "nanodroplet"
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
                                        tuple(map(float, pt)) for pt in corners_arr
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

                    out_path = os.path.join(output_dir, frame_name + ".png")
                    img_out.save(out_path)
                    saved_count += 1

            except Exception as e:
                print(f"Error annotating all-categories raw frame {frame_name}: {e}")

        print(
            f"All-categories raw-frame annotation complete: {output_dir} ({saved_count} images saved)"
        )
