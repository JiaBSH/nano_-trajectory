"""Frame-by-frame segmentation processing."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


class FrameProcessingMixin:
    """Provide frame-by-frame segmentation processing."""

    def process_all_frames(self):
        for frame_id, json_name in enumerate(self.json_files):
            json_path = os.path.join(self.json_dir, json_name)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            frame_name = Path(json_name).stem
            try:
                nm_per_px = self._nm_per_px_for_frame(frame_name)
            except Exception as e:
                print(
                    f"[skip] frame_id={frame_id} frame_name={frame_name}: scale lookup error: {e}"
                )
                continue
            if nm_per_px is None:
                print(
                    f"[skip] frame_id={frame_id} frame_name={frame_name}: no matching scale in CSV"
                )
                continue

            if self.pin_reference_enabled:
                pin_centroid = self._compute_pin_centroid(data)
                if pin_centroid is None:
                    self.skipped_no_pin_frames += 1
                    if self.skip_frames_without_pin:
                        continue
                    shift = self._compute_pin_shift(data)
                else:
                    shift = pin_centroid
                    self.pin_reference_records.append(
                        [
                            int(frame_id),
                            frame_name,
                            float(nm_per_px),
                            float(pin_centroid[0]),
                            float(pin_centroid[1]),
                            float(pin_centroid[0]) * float(nm_per_px),
                            float(pin_centroid[1]) * float(nm_per_px),
                        ]
                    )
            else:
                shift = self._compute_pin_shift(data)

            self._process_target_objects(data, frame_id, frame_name, nm_per_px, shift)
            self.processed_frame_count += 1

        if self.pin_reference_enabled:
            print(
                f"[pin] processed {self.processed_frame_count} frames with pin reference; "
                f"skipped {self.skipped_no_pin_frames} frames without pin."
            )
            if self.max_particle_pin_distance_nm is not None:
                print(
                    f"[pin] filtered {self.filtered_far_particle_count} {self.target_category} objects farther than "
                    f"{self.max_particle_pin_distance_nm:.6f} nm from pin centroid."
                )

    def _compute_pin_centroid(self, data):
        weighted_sum = np.zeros(2, dtype=np.float64)
        total_area = 0.0
        fallback_pts = []
        for obj in data.get("objects", []):
            if obj.get("category") == self.pin_category:
                seg = obj.get("segmentation")
                if seg is None:
                    continue
                pts = np.asarray(seg, dtype=np.float64)
                if pts.ndim == 2 and pts.shape[0] > 0 and pts.shape[1] >= 2:
                    centroid, area = self.polygon_centroid(pts[:, :2])
                    if centroid is None:
                        continue
                    if area > 0:
                        weighted_sum += centroid * area
                        total_area += area
                    else:
                        fallback_pts.append(pts[:, :2])

        if total_area > 0:
            return weighted_sum / total_area

        if len(fallback_pts) > 0:
            return np.vstack(fallback_pts).mean(axis=0)

        return None

    def _compute_pin_shift(self, data):
        pin_centroid = self._compute_pin_centroid(data)
        if pin_centroid is not None:
            if self.ref_pin_centroid is None:
                self.ref_pin_centroid = pin_centroid.copy()

            shift = pin_centroid - self.ref_pin_centroid
            self.last_shift = shift
        else:
            shift = self.last_shift

        return shift

    def _process_target_objects(self, data, frame_id, frame_name, nm_per_px, shift):
        for obj in data.get("objects", []):
            if obj.get("category") != self.target_category:
                continue

            pts = np.array(obj["segmentation"], dtype=np.float32)
            pts = pts - shift  # ★ 去整体漂移

            if pts.shape[0] < 3:
                continue

            # ---- 面积 ----
            area_px2 = self.polygon_area(pts)
            area_nm2 = float(area_px2) * float(nm_per_px) * float(nm_per_px)

            # ---- 质心 ----
            centroid, _centroid_area = self.polygon_centroid(pts)
            if centroid is None:
                centroid = pts.mean(axis=0)
            cx_px, cy_px = float(centroid[0]), float(centroid[1])
            cx_nm, cy_nm = cx_px * float(nm_per_px), cy_px * float(nm_per_px)
            max_pin_dist = self.max_particle_pin_distance_nm
            if self.pin_reference_enabled and max_pin_dist is not None:
                dist_nm = float(np.hypot(cx_nm, cy_nm))
                if dist_nm > float(max_pin_dist):
                    self.filtered_far_particle_count += 1
                    self.filtered_far_particle_records.append(
                        [
                            int(frame_id),
                            frame_name,
                            float(nm_per_px),
                            cx_nm,
                            cy_nm,
                            dist_nm,
                            float(max_pin_dist),
                            area_nm2,
                        ]
                    )
                    continue

            self.area_records.append([frame_id, frame_name, float(nm_per_px), area_nm2])
            self.centroid_records.append(
                [frame_id, frame_name, float(nm_per_px), cx_nm, cy_nm]
            )

            if not self.compute_diameter_height_enabled:
                self.diameter_height_records.append(
                    [frame_id, frame_name, nm_per_px, cx_nm, cy_nm, 0, 0, {}]
                )
            else:
                # ---- Diameter and Height (Rotating Calipers / Minimum Area Rectangle) ----
                # The droplet is a semi-circle projected essentially as a "D" shape.
                # The "bottom" is the flat side of the D.
                # We need to find the orientation of this flat side to measure Diameter (length of flat side)
                # and Height (max perpendicular distance from flat side).
                try:
                    if len(pts) >= 3:
                        d_px, h_px, box_info = self._compute_droplet_dims_oriented(pts)
                        d_nm = d_px * nm_per_px
                        h_nm = h_px * nm_per_px

                        self.diameter_height_records.append(
                            [
                                frame_id,
                                frame_name,
                                nm_per_px,
                                cx_nm,
                                cy_nm,
                                d_nm,
                                h_nm,
                                box_info,
                            ]
                        )
                    else:
                        self.diameter_height_records.append(
                            [frame_id, frame_name, nm_per_px, cx_nm, cy_nm, 0, 0, {}]
                        )
                except Exception as e:
                    # Fallback to AABB
                    print(f"Error in oriented calc: {e}, using AABB")
                    min_x, min_y = pts.min(axis=0)
                    max_x, max_y = pts.max(axis=0)
                    d_nm = (max_x - min_x) * nm_per_px
                    h_nm = (max_y - min_y) * nm_per_px
                    self.diameter_height_records.append(
                        [frame_id, frame_name, nm_per_px, cx_nm, cy_nm, d_nm, h_nm, {}]
                    )

            # ---- 每个目标的聚合记录（用于追踪面积曲线）----
            self.object_records.append(
                [frame_id, frame_name, float(nm_per_px), cx_nm, cy_nm, area_nm2]
            )

            # ---- 轮廓（每帧一行）----
            row = [frame_id, frame_name]
            for x, y in pts:
                x_nm = float(x) * float(nm_per_px)
                y_nm = float(y) * float(nm_per_px)
                row.append(f"({x_nm:.3f},{y_nm:.3f})")
            self.contour_records.append(row)
