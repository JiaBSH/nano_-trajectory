"""CSV result serialization."""

from __future__ import annotations

import csv
import os


class CsvExportMixin:
    """Provide csv result serialization."""

    def _build_export_instance_ids(
        self, max_dist=50.0, id_mode="event", use_display_id=True
    ):
        """Build per-record droplet ids aligned with object_records order."""
        if len(self.object_records) == 0:
            return []

        mode = str(id_mode).strip().lower()
        if mode != "event":
            raise NotImplementedError(
                "export_results currently supports id_mode='event' only"
            )

        series_by_id, assigned_ids_by_frame, _events = (
            self._event_id_series_for_object_records(max_dist=max_dist)
        )

        if bool(use_display_id):
            display_id_of = self._display_id_mapping(series_by_id)
            assigned_ids_by_frame = {
                int(frame_id): [
                    int(display_id_of.get(int(instance_id), int(instance_id)))
                    for instance_id in ids
                ]
                for frame_id, ids in assigned_ids_by_frame.items()
            }

        export_ids = []
        for frame_id in sorted(assigned_ids_by_frame.keys()):
            export_ids.extend(assigned_ids_by_frame[frame_id])

        if len(export_ids) != len(self.object_records):
            raise ValueError(
                f"Export instance-id count mismatch: ids={len(export_ids)} object_records={len(self.object_records)}"
            )

        return export_ids

    def export_results(self, max_dist=50.0, id_mode="event", use_display_id=True):
        export_ids = self._build_export_instance_ids(
            max_dist=max_dist,
            id_mode=id_mode,
            use_display_id=use_display_id,
        )

        # 面积
        path1 = os.path.join(
            self.output_root, f"{self.target_category}_area_vs_frame.csv"
        )
        with open(path1, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["instance_id", "frame_id", "frame_name", "nm_per_pixel", "area_nm2"]
            )
            writer.writerows(
                [
                    [
                        int(instance_id),
                        frame_id,
                        frame_name,
                        f"{nm_per_px:.6f}",
                        f"{area_nm2:.6f}",
                    ]
                    for instance_id, (frame_id, frame_name, nm_per_px, area_nm2) in zip(
                        export_ids, self.area_records
                    )
                ]
            )

        # 轮廓（每帧一行）
        path2 = os.path.join(
            self.output_root, f"{self.target_category}_contours_by_frame.csv"
        )
        with open(path2, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["instance_id", "frame_id", "frame_name", "contour_points_nm"]
            )
            writer.writerows(
                [
                    [int(instance_id)] + row
                    for instance_id, row in zip(export_ids, self.contour_records)
                ]
            )

        # 质心
        path3 = os.path.join(self.output_root, f"{self.target_category}_centroids.csv")
        with open(path3, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "instance_id",
                    "frame_id",
                    "frame_name",
                    "nm_per_pixel",
                    "cx_nm",
                    "cy_nm",
                ]
            )
            writer.writerows(
                [
                    [
                        int(instance_id),
                        frame_id,
                        frame_name,
                        f"{nm_per_px:.6f}",
                        f"{cx_nm:.6f}",
                        f"{cy_nm:.6f}",
                    ]
                    for instance_id, (
                        frame_id,
                        frame_name,
                        nm_per_px,
                        cx_nm,
                        cy_nm,
                    ) in zip(export_ids, self.centroid_records)
                ]
            )

        # Diameter and Height
        path4 = os.path.join(
            self.output_root, f"{self.target_category}_diameter_height_vs_frame.csv"
        )
        with open(path4, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "instance_id",
                    "frame_id",
                    "frame_name",
                    "nm_per_pixel",
                    "cx_nm",
                    "cy_nm",
                    "diameter_nm",
                    "height_nm",
                ]
            )
            for instance_id, row in zip(export_ids, self.diameter_height_records):
                # row structure: [frame_id, frame_name, nm_per_px, cx_nm, cy_nm, d_nm, h_nm, min_x, min_y, max_x, max_y]
                # we only export the first 7 fields here
                writer.writerow(
                    [
                        int(instance_id),
                        row[0],
                        row[1],
                        f"{row[2]:.6f}",
                        f"{row[3]:.6f}",
                        f"{row[4]:.6f}",
                        f"{row[5]:.6f}",
                        f"{row[6]:.6f}",
                    ]
                )

        path5 = None
        if self.pin_reference_enabled:
            path5 = os.path.join(
                self.output_root, f"{self.target_category}_pin_reference_centroids.csv"
            )
            with open(path5, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "frame_id",
                        "frame_name",
                        "nm_per_pixel",
                        "pin_cx_px",
                        "pin_cy_px",
                        "pin_cx_nm",
                        "pin_cy_nm",
                    ]
                )
                for row in self.pin_reference_records:
                    (
                        frame_id,
                        frame_name,
                        nm_per_px,
                        pin_cx_px,
                        pin_cy_px,
                        pin_cx_nm,
                        pin_cy_nm,
                    ) = row
                    writer.writerow(
                        [
                            int(frame_id),
                            frame_name,
                            f"{float(nm_per_px):.6f}",
                            f"{float(pin_cx_px):.6f}",
                            f"{float(pin_cy_px):.6f}",
                            f"{float(pin_cx_nm):.6f}",
                            f"{float(pin_cy_nm):.6f}",
                        ]
                    )

        path6 = None
        if self.pin_reference_enabled and self.max_particle_pin_distance_nm is not None:
            path6 = os.path.join(
                self.output_root, f"{self.target_category}_filtered_far_from_pin.csv"
            )
            with open(path6, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "frame_id",
                        "frame_name",
                        "nm_per_pixel",
                        "cx_nm",
                        "cy_nm",
                        "distance_to_pin_nm",
                        "threshold_nm",
                        "area_nm2",
                    ]
                )
                for row in self.filtered_far_particle_records:
                    (
                        frame_id,
                        frame_name,
                        nm_per_px,
                        cx_nm,
                        cy_nm,
                        dist_nm,
                        threshold_nm,
                        area_nm2,
                    ) = row
                    writer.writerow(
                        [
                            int(frame_id),
                            frame_name,
                            f"{float(nm_per_px):.6f}",
                            f"{float(cx_nm):.6f}",
                            f"{float(cy_nm):.6f}",
                            f"{float(dist_nm):.6f}",
                            f"{float(threshold_nm):.6f}",
                            f"{float(area_nm2):.6f}",
                        ]
                    )

        distance_paths = []
        if self.compute_boundary_distances_enabled:
            particle_ids_by_frame = self._tracked_category_ids(
                self.boundary_particle_records,
                max_dist=max_dist,
                category=self.particle_category,
            )
            droplet_ids_by_frame = self._tracked_category_ids(
                self.boundary_droplet_records,
                max_dist=max_dist,
                category=self.droplet_category,
            )

            def tracked_label(ids_by_frame, frame_id, frame_index, prefix):
                ids = ids_by_frame.get(int(frame_id), [])
                zero_based_index = int(frame_index) - 1
                if zero_based_index < 0 or zero_based_index >= len(ids):
                    raise ValueError(
                        f"Missing tracked {prefix} ID for frame={frame_id}, "
                        f"frame_index={frame_index}"
                    )
                return f"{prefix}{int(ids[zero_based_index])}"

            particle_particle_path = os.path.join(
                self.output_root,
                f"{self.particle_category}_to_{self.particle_category}_boundary_distances.csv",
            )
            with open(particle_particle_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "frame_id",
                        "frame_name",
                        "nm_per_pixel",
                        "particle_1_id",
                        "particle_2_id",
                        "boundary_distance_nm",
                    ]
                )
                for row in self.particle_particle_distance_records:
                    pair_ids = sorted(
                        (
                            tracked_label(particle_ids_by_frame, row[0], row[3], "P"),
                            tracked_label(particle_ids_by_frame, row[0], row[5], "P"),
                        ),
                        key=lambda value: int(value[1:]),
                    )
                    writer.writerow(
                        [
                            int(row[0]),
                            row[1],
                            f"{float(row[2]):.6f}",
                            pair_ids[0],
                            pair_ids[1],
                            f"{float(row[7]):.6f}",
                        ]
                    )
            distance_paths.append(particle_particle_path)

            particle_droplet_path = os.path.join(
                self.output_root,
                f"{self.particle_category}_to_{self.droplet_category}_boundary_distances.csv",
            )
            with open(particle_droplet_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "frame_id",
                        "frame_name",
                        "nm_per_pixel",
                        "particle_id",
                        "droplet_id",
                        "boundary_distance_nm",
                    ]
                )
                for row in self.particle_droplet_distance_records:
                    writer.writerow(
                        [
                            int(row[0]),
                            row[1],
                            f"{float(row[2]):.6f}",
                            tracked_label(particle_ids_by_frame, row[0], row[3], "P"),
                            tracked_label(droplet_ids_by_frame, row[0], row[5], "D"),
                            f"{float(row[7]):.6f}",
                        ]
                    )
            distance_paths.append(particle_droplet_path)

            nearest_path = os.path.join(
                self.output_root,
                f"{self.particle_category}_nearest_boundary_distances.csv",
            )
            with open(nearest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "frame_id",
                        "frame_name",
                        "nm_per_pixel",
                        "particle_id",
                        "nearest_particle_id",
                        "nearest_particle_boundary_distance_nm",
                        "nearest_droplet_id",
                        "nearest_droplet_boundary_distance_nm",
                    ]
                )
                for row in self.particle_nearest_distance_records:
                    writer.writerow(
                        [
                            int(row[0]),
                            row[1],
                            f"{float(row[2]):.6f}",
                            tracked_label(particle_ids_by_frame, row[0], row[3], "P"),
                            (
                                ""
                                if row[5] is None
                                else tracked_label(
                                    particle_ids_by_frame, row[0], row[5], "P"
                                )
                            ),
                            "" if row[7] is None else f"{float(row[7]):.6f}",
                            (
                                ""
                                if row[8] is None
                                else tracked_label(
                                    droplet_ids_by_frame, row[0], row[8], "D"
                                )
                            ),
                            "" if row[10] is None else f"{float(row[10]):.6f}",
                        ]
                    )
            distance_paths.append(nearest_path)

        print("Export finished:")
        print(f" - {path1}")
        print(f" - {path2}")
        print(f" - {path3}")
        print(f" - {path4}")
        if path5 is not None:
            print(f" - {path5}")
        if path6 is not None:
            print(f" - {path6}")
        for distance_path in distance_paths:
            print(f" - {distance_path}")

    def export_tracked_area_results(self, tracks, out_csv=None):
        """Export tracked area series.

        CSV columns: track_id, frame_id, frame_name, nm_per_pixel, area_nm2, cx_nm, cy_nm
        """
        if out_csv is None:
            out_csv = os.path.join(
                self.output_root, f"{self.target_category}_tracked_area_vs_frame.csv"
            )
        elif not os.path.isabs(out_csv):
            out_csv = os.path.join(self.output_root, out_csv)

        rows = []
        for track_id, t in enumerate(tracks):
            for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in t["points"]:
                rows.append(
                    [
                        track_id,
                        frame_id,
                        frame_name,
                        f"{nm_per_px:.6f}",
                        f"{area_nm2:.6f}",
                        f"{cx_nm:.6f}",
                        f"{cy_nm:.6f}",
                    ]
                )

        rows.sort(key=lambda r: (r[0], r[1]))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "track_id",
                    "frame_id",
                    "frame_name",
                    "nm_per_pixel",
                    "area_nm2",
                    "cx_nm",
                    "cy_nm",
                ]
            )
            writer.writerows(rows)

        print(f" - {out_csv}")

    def export_id_series(self, series_by_id, out_csv=None, display_id_of=None):
        """Export area series keyed by a globally-incrementing instance id.

        CSV columns: instance_id, frame_id, frame_name, nm_per_pixel, area_nm2, cx_nm, cy_nm
        """
        if out_csv is None:
            out_csv = os.path.join(
                self.output_root, f"{self.target_category}_instance_area_vs_frame.csv"
            )
        elif not os.path.isabs(out_csv):
            out_csv = os.path.join(self.output_root, out_csv)

        rows = []
        for instance_id, points in series_by_id.items():
            exported_id = int(
                instance_id
                if display_id_of is None
                else display_id_of.get(int(instance_id), int(instance_id))
            )
            for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in points:
                rows.append(
                    [
                        exported_id,
                        int(frame_id),
                        frame_name,
                        f"{float(nm_per_px):.6f}",
                        f"{float(area_nm2):.6f}",
                        f"{float(cx_nm):.6f}",
                        f"{float(cy_nm):.6f}",
                    ]
                )

        rows.sort(key=lambda r: (r[0], r[1]))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "instance_id",
                    "frame_id",
                    "frame_name",
                    "nm_per_pixel",
                    "area_nm2",
                    "cx_nm",
                    "cy_nm",
                ]
            )
            writer.writerows(rows)

        print(f" - {out_csv}")

    def export_speed_series(
        self, speed_series_by_id, out_csv=None, display_id_of=None
    ):
        """Export per-instance speed series (from centroid displacement).

        Speed is computed between consecutive detections of the same instance:
            speed = distance_nm / (delta_frame * frame_interval_s)

        CSV columns: instance_id, frame_id, frame_name, speed_nm_per_s
        """
        if out_csv is None:
            out_csv = os.path.join(
                self.output_root, f"{self.target_category}_instance_speed_vs_frame.csv"
            )
        elif not os.path.isabs(out_csv):
            out_csv = os.path.join(self.output_root, out_csv)

        rows = []
        for instance_id, points in speed_series_by_id.items():
            exported_id = int(
                instance_id
                if display_id_of is None
                else display_id_of.get(int(instance_id), int(instance_id))
            )
            for frame_id, frame_name, speed_nm_per_s in points:
                rows.append(
                    [
                        exported_id,
                        int(frame_id),
                        frame_name,
                        f"{float(speed_nm_per_s):.6f}",
                    ]
                )

        rows.sort(key=lambda r: (r[0], r[1]))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "frame_id", "frame_name", "speed_nm_per_s"])
            writer.writerows(rows)

        print(f" - {out_csv}")
