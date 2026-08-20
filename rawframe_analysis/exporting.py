"""CSV result serialization."""

from __future__ import annotations

import csv
import os
import re


class CsvExportMixin:
    """Provide csv result serialization."""

    def _export_instance_label(self, instance_id):
        """Return the same compact ID used by pair annotations and plots."""
        return self._format_category_display_id(int(instance_id))

    @staticmethod
    def _csv_object_key(row):
        """Return the cross-table key shared by per-instance result rows."""
        return (
            str(row.get("instance_id", "")),
            str(row.get("frame_id", "")),
            str(row.get("frame_name", "")),
        )

    def validate_exported_csv_ids(self):
        """Fail the run if any exported CSV uses an inconsistent object ID.

        Frame-level aggregate/reference tables intentionally have no object ID.
        Every object-level ID column is checked, and the complete target tables
        must contain exactly the same (instance_id, frame_id, frame_name) keys.
        """
        target_prefix = self._category_id_prefix()
        id_prefixes = {
            "instance_id": target_prefix,
            "particle_id": "P",
            "particle_1_id": "P",
            "particle_2_id": "P",
            "nearest_particle_id": "P",
            "droplet_id": "D",
            "nearest_droplet_id": "D",
        }
        csv_rows = {}
        csv_fields = {}
        checked_columns = []
        for filename in sorted(os.listdir(self.output_root)):
            if not filename.lower().endswith(".csv"):
                continue
            path = os.path.join(self.output_root, filename)
            with open(path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = list(reader.fieldnames or [])
            csv_rows[filename] = rows
            csv_fields[filename] = fieldnames
            for column, prefix in id_prefixes.items():
                if column not in fieldnames:
                    continue
                checked_columns.append((filename, column))
                pattern = re.compile(rf"^{re.escape(prefix)}[1-9][0-9]*$")
                invalid = sorted(
                    {
                        str(row.get(column, ""))
                        for row in rows
                        if str(row.get(column, ""))
                        and pattern.fullmatch(str(row.get(column, ""))) is None
                    }
                )
                if invalid:
                    raise ValueError(
                        f"CSV ID validation failed: {filename}.{column} "
                        f"contains invalid values {invalid[:10]}"
                    )

        object_file_patterns = (
            f"{self.target_category}_area_vs_frame.csv",
            f"{self.target_category}_centroids.csv",
            f"{self.target_category}_contours_by_frame.csv",
            f"{self.target_category}_diameter_height_vs_frame.csv",
            f"{self.target_category}_filtered_far_from_pin.csv",
            f"{self.target_category}_instance_area_vs_frame.csv",
            f"{self.target_category}_tracked_area_vs_frame.csv",
        )
        for filename in csv_rows:
            is_speed = filename.startswith(
                f"{self.target_category}_instance_speed_"
            ) and filename.endswith(".csv")
            if (
                filename in object_file_patterns or is_speed
            ) and "instance_id" not in csv_fields[filename]:
                raise ValueError(
                    f"CSV ID validation failed: {filename} is an object-level "
                    "table but has no instance_id column"
                )

        complete_files = [
            f"{self.target_category}_area_vs_frame.csv",
            f"{self.target_category}_centroids.csv",
            f"{self.target_category}_contours_by_frame.csv",
            f"{self.target_category}_diameter_height_vs_frame.csv",
            f"{self.target_category}_instance_area_vs_frame.csv",
        ]
        present_complete = [name for name in complete_files if name in csv_rows]
        if present_complete:
            canonical_name = present_complete[0]
            canonical = {
                self._csv_object_key(row) for row in csv_rows[canonical_name]
            }
            for filename in present_complete[1:]:
                candidate = {
                    self._csv_object_key(row) for row in csv_rows[filename]
                }
                if candidate != canonical:
                    raise ValueError(
                        f"CSV ID validation failed: {filename} object/frame keys "
                        f"do not match {canonical_name}"
                    )

            speed_names = [
                name
                for name in csv_rows
                if name.startswith(f"{self.target_category}_instance_speed_")
                and name.endswith(".csv")
            ]
            for filename in speed_names:
                speed_keys = {
                    self._csv_object_key(row) for row in csv_rows[filename]
                }
                if not speed_keys.issubset(canonical):
                    raise ValueError(
                        f"CSV ID validation failed: {filename} contains "
                        "object/frame keys absent from the complete target tables"
                    )

        print(
            f"[validate] CSV object IDs passed: files={len(csv_rows)}, "
            f"id_columns={len(checked_columns)}"
        )
        return {
            "files": len(csv_rows),
            "id_columns": len(checked_columns),
            "complete_tables": len(present_complete),
        }

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
                        self._export_instance_label(instance_id),
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
                    [self._export_instance_label(instance_id)] + row
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
                        self._export_instance_label(instance_id),
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
                        self._export_instance_label(instance_id),
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
            filtered_ids_by_frame = {}
            if self.filtered_far_particle_records:
                normalized_target = str(self.target_category).strip().lower()
                if normalized_target == str(self.particle_category).strip().lower():
                    target_tracking_records = self.boundary_particle_records
                elif normalized_target == str(self.droplet_category).strip().lower():
                    target_tracking_records = self.boundary_droplet_records
                else:
                    target_tracking_records = []
                if not target_tracking_records:
                    raise ValueError(
                        "Cannot export consistent IDs for filtered_far_from_pin: "
                        "enable boundary-distance processing for the target category"
                    )
                filtered_ids_by_frame = self._tracked_category_ids(
                    target_tracking_records,
                    max_dist=max_dist,
                    category=self.target_category,
                )
            with open(path6, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "instance_id",
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
                        frame_instance_index,
                    ) = row
                    frame_ids = filtered_ids_by_frame.get(int(frame_id), [])
                    zero_based_index = int(frame_instance_index) - 1
                    if zero_based_index < 0 or zero_based_index >= len(frame_ids):
                        raise ValueError(
                            "Missing annotated instance ID for filtered object: "
                            f"frame={frame_id}, frame_index={frame_instance_index}"
                        )
                    writer.writerow(
                        [
                            self._format_category_display_id(
                                frame_ids[zero_based_index], self.target_category
                            ),
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

        CSV columns: instance_id, frame_id, frame_name, nm_per_pixel, area_nm2, cx_nm, cy_nm
        """
        if out_csv is None:
            out_csv = os.path.join(
                self.output_root, f"{self.target_category}_tracked_area_vs_frame.csv"
            )
        elif not os.path.isabs(out_csv):
            out_csv = os.path.join(self.output_root, out_csv)

        rows = []
        for track_id, t in enumerate(tracks, start=1):
            instance_label = self._export_instance_label(track_id)
            for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in t["points"]:
                rows.append(
                    [
                        instance_label,
                        frame_id,
                        frame_name,
                        f"{nm_per_px:.6f}",
                        f"{area_nm2:.6f}",
                        f"{cx_nm:.6f}",
                        f"{cy_nm:.6f}",
                    ]
                )

        prefix = self._category_id_prefix()
        rows.sort(key=lambda r: (int(str(r[0])[len(prefix) :]), r[1]))
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
            exported_label = self._export_instance_label(exported_id)
            for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in points:
                rows.append(
                    [
                        exported_label,
                        int(frame_id),
                        frame_name,
                        f"{float(nm_per_px):.6f}",
                        f"{float(area_nm2):.6f}",
                        f"{float(cx_nm):.6f}",
                        f"{float(cy_nm):.6f}",
                    ]
                )

        prefix = self._category_id_prefix()
        rows.sort(key=lambda r: (int(str(r[0])[len(prefix) :]), r[1]))
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
            exported_label = self._export_instance_label(exported_id)
            for frame_id, frame_name, speed_nm_per_s in points:
                rows.append(
                    [
                        exported_label,
                        int(frame_id),
                        frame_name,
                        f"{float(speed_nm_per_s):.6f}",
                    ]
                )

        prefix = self._category_id_prefix()
        rows.sort(key=lambda r: (int(str(r[0])[len(prefix) :]), r[1]))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "frame_id", "frame_name", "speed_nm_per_s"])
            writer.writerows(rows)

        print(f" - {out_csv}")
