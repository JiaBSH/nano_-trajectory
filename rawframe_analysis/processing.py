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
            data = self._postprocess_frame_instances(data, frame_name=frame_name)
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
                raw_pin_centroid = self._compute_pin_centroid(data)
                if raw_pin_centroid is None:
                    self.skipped_no_pin_frames += 1
                    if self.skip_frames_without_pin:
                        continue
                    shift = (
                        self.stabilized_pin_centroid.copy()
                        if self.stabilized_pin_centroid is not None
                        else np.zeros(2, dtype=np.float64)
                    )
                else:
                    pin_centroid = self._stabilize_pin_centroid(raw_pin_centroid)
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
            self._process_boundary_distances(
                data, frame_id, frame_name, nm_per_px, shift
            )
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
        if self.instance_overlap_postprocess_enabled:
            print(
                f"[postprocess] merged {self.same_category_suppressed_count} additional "
                "same-category masks; cross-category overlaps were preserved."
            )

    def _category_polygons(self, data, category, shift):
        """Return valid category polygons with frame-local and JSON indices."""
        polygons = []
        for json_object_index, obj in enumerate(data.get("objects", [])):
            if obj.get("category") != category:
                continue
            points = np.asarray(obj.get("segmentation"), dtype=np.float64)
            if (
                points.ndim != 2
                or points.shape[0] < 3
                or points.shape[1] < 2
                or not np.all(np.isfinite(points[:, :2]))
            ):
                continue
            tracking_points = points[:, :2] - np.asarray(shift, dtype=np.float64)
            centroid, area_px2 = self.polygon_centroid(tracking_points)
            if centroid is None:
                centroid = tracking_points.mean(axis=0)
            polygons.append(
                {
                    "frame_index": len(polygons) + 1,
                    "json_object_index": int(
                        obj.get("_raw_json_object_index", json_object_index)
                    ),
                    "points": points[:, :2],
                    "tracking_centroid_px": centroid,
                    "area_px2": float(area_px2),
                }
            )
        return polygons

    def _process_boundary_distances(self, data, frame_id, frame_name, nm_per_px, shift):
        if not self.compute_boundary_distances_enabled:
            return

        particles = self._category_polygons(data, self.particle_category, shift)
        droplets = self._category_polygons(data, self.droplet_category, shift)
        nearest_particles = [None] * len(particles)
        nearest_droplets = [None] * len(particles)
        scale = float(nm_per_px)

        for record_buffer, objects in (
            (self.boundary_particle_records, particles),
            (self.boundary_droplet_records, droplets),
        ):
            for obj in objects:
                centroid = obj["tracking_centroid_px"]
                record_buffer.append(
                    [
                        int(frame_id),
                        frame_name,
                        scale,
                        obj["frame_index"],
                        obj["json_object_index"],
                        float(centroid[0]) * scale,
                        float(centroid[1]) * scale,
                        float(obj["area_px2"]) * scale * scale,
                    ]
                )

        for first_index, first in enumerate(particles):
            for second_index in range(first_index + 1, len(particles)):
                second = particles[second_index]
                distance_nm = (
                    self.polygon_boundary_distance(first["points"], second["points"])
                    * scale
                )
                self.particle_particle_distance_records.append(
                    [
                        int(frame_id),
                        frame_name,
                        scale,
                        first["frame_index"],
                        first["json_object_index"],
                        second["frame_index"],
                        second["json_object_index"],
                        float(distance_nm),
                    ]
                )
                candidate_for_first = (float(distance_nm), second)
                candidate_for_second = (float(distance_nm), first)
                if (
                    nearest_particles[first_index] is None
                    or candidate_for_first[0] < nearest_particles[first_index][0]
                ):
                    nearest_particles[first_index] = candidate_for_first
                if (
                    nearest_particles[second_index] is None
                    or candidate_for_second[0] < nearest_particles[second_index][0]
                ):
                    nearest_particles[second_index] = candidate_for_second

        for particle_index, particle in enumerate(particles):
            for droplet in droplets:
                distance_nm = (
                    self.polygon_boundary_distance(
                        particle["points"], droplet["points"]
                    )
                    * scale
                )
                self.particle_droplet_distance_records.append(
                    [
                        int(frame_id),
                        frame_name,
                        scale,
                        particle["frame_index"],
                        particle["json_object_index"],
                        droplet["frame_index"],
                        droplet["json_object_index"],
                        float(distance_nm),
                    ]
                )
                candidate = (float(distance_nm), droplet)
                if (
                    nearest_droplets[particle_index] is None
                    or candidate[0] < nearest_droplets[particle_index][0]
                ):
                    nearest_droplets[particle_index] = candidate

        for particle_index, particle in enumerate(particles):
            nearest_particle = nearest_particles[particle_index]
            nearest_droplet = nearest_droplets[particle_index]
            self.particle_nearest_distance_records.append(
                [
                    int(frame_id),
                    frame_name,
                    scale,
                    particle["frame_index"],
                    particle["json_object_index"],
                    (
                        None
                        if nearest_particle is None
                        else nearest_particle[1]["frame_index"]
                    ),
                    (
                        None
                        if nearest_particle is None
                        else nearest_particle[1]["json_object_index"]
                    ),
                    None if nearest_particle is None else nearest_particle[0],
                    (
                        None
                        if nearest_droplet is None
                        else nearest_droplet[1]["frame_index"]
                    ),
                    (
                        None
                        if nearest_droplet is None
                        else nearest_droplet[1]["json_object_index"]
                    ),
                    None if nearest_droplet is None else nearest_droplet[0],
                ]
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

    def _stabilize_pin_centroid(self, pin_centroid):
        """Return a causal EMA-smoothed pin centroid in pixel coordinates.

        The first valid observation initializes the filter exactly.  Keeping the
        state in pixels makes the same stabilized reference available to every
        downstream coordinate calculation, even when the physical scale changes
        between frames.
        """
        observation = np.asarray(pin_centroid, dtype=np.float64)
        if self.stabilized_pin_centroid is None:
            self.stabilized_pin_centroid = observation.copy()
        else:
            alpha = float(self.pin_centroid_smoothing_alpha)
            self.stabilized_pin_centroid += alpha * (
                observation - self.stabilized_pin_centroid
            )
        return self.stabilized_pin_centroid.copy()

    def _compute_pin_shift(self, data):
        pin_centroid = self._compute_pin_centroid(data)
        if pin_centroid is not None:
            pin_centroid = self._stabilize_pin_centroid(pin_centroid)
            if self.ref_pin_centroid is None:
                self.ref_pin_centroid = pin_centroid.copy()

            shift = pin_centroid - self.ref_pin_centroid
            self.last_shift = shift
        else:
            shift = self.last_shift

        return shift

    def _process_target_objects(self, data, frame_id, frame_name, nm_per_px, shift):
        target_frame_index = 0
        for obj in data.get("objects", []):
            if obj.get("category") != self.target_category:
                continue

            pts = np.array(obj["segmentation"], dtype=np.float32)
            pts = pts - shift  # ★ 去整体漂移

            if pts.shape[0] < 3:
                continue
            target_frame_index += 1

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
                            int(target_frame_index),
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
