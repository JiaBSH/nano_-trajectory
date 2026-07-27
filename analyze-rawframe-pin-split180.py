import argparse
import csv
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_base_tracker():
    base_path = Path(__file__).resolve().parent / "analyze-rawframe.py"
    spec = importlib.util.spec_from_file_location("analyze_rawframe_base", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base tracker from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GasTracker


GasTracker = _load_base_tracker()


class PinSplitGasTracker(GasTracker):
    """GasTracker variant that compares detections left/right of pin_x + offset."""

    def __init__(self, *args, split_offset_px=-180.0, output_root=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_offset_px = float(split_offset_px)

        if output_root is None:
            offset_label = self._offset_label()
            output_root = f"{self.gas_category}_pin_split{offset_label}"
        self.output_root = os.fspath(output_root)
        os.makedirs(self.output_root, exist_ok=True)

        self.side_records = {
            "left": self._empty_side_records(),
            "right": self._empty_side_records(),
        }
        self.split_records = []
        self.pin_line_records = []
        self._line_by_frame = {}

    def _offset_label(self):
        magnitude = abs(float(self.split_offset_px))
        if magnitude.is_integer():
            amount = str(int(magnitude))
        else:
            amount = str(magnitude).replace(".", "p")
        if self.split_offset_px < 0:
            return f"_left{amount}"
        if self.split_offset_px > 0:
            return f"_right{amount}"
        return "_at_pin"

    def _offset_display_label(self):
        if self.split_offset_px < 0:
            return f"pin - {abs(float(self.split_offset_px)):g} px"
        if self.split_offset_px > 0:
            return f"pin + {float(self.split_offset_px):g} px"
        return "pin"

    @staticmethod
    def _empty_side_records():
        return {
            "area_records": [],
            "contour_records": [],
            "centroid_records": [],
            "object_records": [],
            "diameter_height_records": [],
        }

    def process_all_frames(self, max_frames=None):
        json_files = self.json_files
        if max_frames is not None:
            json_files = json_files[: max(0, int(max_frames))]

        for frame_id, json_name in enumerate(json_files):
            json_path = os.path.join(self.json_dir, json_name)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            frame_name = Path(json_name).stem
            try:
                nm_per_px = self._nm_per_px_for_frame(frame_name)
            except Exception as e:
                print(f"[skip] frame_id={frame_id} frame_name={frame_name}: scale lookup error: {e}")
                continue
            if nm_per_px is None:
                print(f"[skip] frame_id={frame_id} frame_name={frame_name}: no matching scale in CSV")
                continue

            shift, line_info = self._compute_pin_shift_and_split_line(data, frame_id, frame_name, nm_per_px)
            self._process_gas_objects_with_split(data, frame_id, frame_name, nm_per_px, shift, line_info)

    def _compute_pin_shift_and_split_line(self, data, frame_id, frame_name, nm_per_px):
        pin_pts = []
        for obj in data.get("objects", []):
            if obj.get("category") == self.pin_category:
                pts = np.array(obj.get("segmentation", []), dtype=np.float32)
                if pts.shape[0] > 0:
                    pin_pts.append(pts)

        status = "missing"
        pin_centroid = None
        if pin_pts:
            pin_pts = np.vstack(pin_pts)
            pin_centroid = pin_pts.mean(axis=0)
            if self.ref_pin_centroid is None:
                self.ref_pin_centroid = pin_centroid.copy()
            shift = pin_centroid - self.ref_pin_centroid
            self.last_shift = shift
            status = "detected"
        else:
            shift = self.last_shift
            if self.ref_pin_centroid is not None:
                pin_centroid = self.ref_pin_centroid + shift
                status = "estimated_from_last_shift"

        line_info = {
            "frame_id": int(frame_id),
            "frame_name": str(frame_name),
            "nm_per_px": float(nm_per_px),
            "pin_status": status,
            "pin_cx_px": "",
            "pin_cy_px": "",
            "split_x_raw_px": "",
            "split_x_aligned_px": "",
            "split_x_aligned_nm": "",
        }

        if pin_centroid is not None:
            split_x_raw = float(pin_centroid[0]) + float(self.split_offset_px)
            split_x_aligned = split_x_raw - float(shift[0])
            line_info.update(
                {
                    "pin_cx_px": float(pin_centroid[0]),
                    "pin_cy_px": float(pin_centroid[1]),
                    "split_x_raw_px": split_x_raw,
                    "split_x_aligned_px": split_x_aligned,
                    "split_x_aligned_nm": split_x_aligned * float(nm_per_px),
                }
            )

        self.pin_line_records.append(line_info)
        self._line_by_frame[int(frame_id)] = line_info
        return shift, line_info

    def _process_gas_objects_with_split(self, data, frame_id, frame_name, nm_per_px, shift, line_info):
        split_x_raw = line_info.get("split_x_raw_px")
        has_line = split_x_raw not in ("", None)
        split_x_raw_f = float(split_x_raw) if has_line else None

        for obj in data.get("objects", []):
            if obj.get("category") != self.gas_category:
                continue

            pts_raw = np.array(obj.get("segmentation", []), dtype=np.float32)
            if pts_raw.shape[0] < 3:
                continue

            pts = pts_raw - shift
            area_px2 = self.polygon_area(pts)
            area_nm2 = float(area_px2) * float(nm_per_px) * float(nm_per_px)

            centroid_raw = pts_raw.mean(axis=0)
            centroid = pts.mean(axis=0)
            cx_raw_px, cy_raw_px = float(centroid_raw[0]), float(centroid_raw[1])
            cx_px, cy_px = float(centroid[0]), float(centroid[1])
            cx_nm, cy_nm = cx_px * float(nm_per_px), cy_px * float(nm_per_px)

            d_nm, h_nm, box_info = self._diameter_height_nm(pts, nm_per_px)

            row = [frame_id, frame_name]
            for x, y in pts:
                row.append(f"({float(x) * float(nm_per_px):.3f},{float(y) * float(nm_per_px):.3f})")

            object_record = [frame_id, frame_name, float(nm_per_px), cx_nm, cy_nm, area_nm2]
            area_record = [frame_id, frame_name, float(nm_per_px), area_nm2]
            centroid_record = [frame_id, frame_name, float(nm_per_px), cx_nm, cy_nm]
            diameter_height_record = [frame_id, frame_name, float(nm_per_px), cx_nm, cy_nm, d_nm, h_nm, box_info]

            self.area_records.append(area_record)
            self.centroid_records.append(centroid_record)
            self.diameter_height_records.append(diameter_height_record)
            self.object_records.append(object_record)
            self.contour_records.append(row)

            side = "unknown"
            signed_distance_px = ""
            signed_distance_nm = ""
            left_area_nm2 = ""
            right_area_nm2 = ""
            crosses_split_line = False
            if has_line:
                side = "left" if cx_raw_px < split_x_raw_f else "right"
                signed_distance_px = cx_raw_px - split_x_raw_f
                signed_distance_nm = float(signed_distance_px) * float(nm_per_px)
                min_x = float(np.min(pts_raw[:, 0]))
                max_x = float(np.max(pts_raw[:, 0]))
                crosses_split_line = min_x < split_x_raw_f < max_x
                left_area_nm2 = self._clipped_area_nm2(pts_raw, split_x_raw_f, keep_left=True, nm_per_px=nm_per_px)
                right_area_nm2 = self._clipped_area_nm2(pts_raw, split_x_raw_f, keep_left=False, nm_per_px=nm_per_px)

                side_bucket = self.side_records[side]
                side_bucket["area_records"].append(area_record)
                side_bucket["centroid_records"].append(centroid_record)
                side_bucket["diameter_height_records"].append(diameter_height_record)
                side_bucket["object_records"].append(object_record)
                side_bucket["contour_records"].append(row)

            self.split_records.append(
                {
                    "frame_id": int(frame_id),
                    "frame_name": str(frame_name),
                    "side_by_centroid": side,
                    "nm_per_px": float(nm_per_px),
                    "pin_status": line_info.get("pin_status", ""),
                    "pin_cx_px": line_info.get("pin_cx_px", ""),
                    "pin_cy_px": line_info.get("pin_cy_px", ""),
                    "split_x_raw_px": line_info.get("split_x_raw_px", ""),
                    "split_x_aligned_px": line_info.get("split_x_aligned_px", ""),
                    "split_x_aligned_nm": line_info.get("split_x_aligned_nm", ""),
                    "cx_raw_px": cx_raw_px,
                    "cy_raw_px": cy_raw_px,
                    "cx_aligned_nm": cx_nm,
                    "cy_aligned_nm": cy_nm,
                    "signed_distance_to_line_px": signed_distance_px,
                    "signed_distance_to_line_nm": signed_distance_nm,
                    "area_nm2": area_nm2,
                    "clipped_left_area_nm2": left_area_nm2,
                    "clipped_right_area_nm2": right_area_nm2,
                    "crosses_split_line": int(bool(crosses_split_line)),
                    "diameter_nm": d_nm,
                    "height_nm": h_nm,
                }
            )

    def _diameter_height_nm(self, pts, nm_per_px):
        try:
            d_px, h_px, box_info = self._compute_droplet_dims_oriented(pts)
            return float(d_px) * float(nm_per_px), float(h_px) * float(nm_per_px), box_info
        except Exception as e:
            print(f"Error in oriented calc: {e}, using AABB")
            min_x, min_y = pts.min(axis=0)
            max_x, max_y = pts.max(axis=0)
            d_nm = float(max_x - min_x) * float(nm_per_px)
            h_nm = float(max_y - min_y) * float(nm_per_px)
            return d_nm, h_nm, {}

    @staticmethod
    def _clip_polygon_vertical(pts, x_line, keep_left=True):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return np.empty((0, 2), dtype=np.float64)

        eps = 1e-9

        def inside(p):
            return p[0] <= x_line + eps if keep_left else p[0] >= x_line - eps

        def intersect(a, b):
            dx = b[0] - a[0]
            if abs(dx) < eps:
                return np.array([x_line, a[1]], dtype=np.float64)
            t = (x_line - a[0]) / dx
            return np.array([x_line, a[1] + t * (b[1] - a[1])], dtype=np.float64)

        output = []
        prev = pts[-1]
        prev_inside = inside(prev)
        for curr in pts:
            curr_inside = inside(curr)
            if curr_inside:
                if not prev_inside:
                    output.append(intersect(prev, curr))
                output.append(curr)
            elif prev_inside:
                output.append(intersect(prev, curr))
            prev = curr
            prev_inside = curr_inside

        if len(output) < 3:
            return np.empty((0, 2), dtype=np.float64)
        return np.vstack(output)

    def _clipped_area_nm2(self, pts_raw, x_line, keep_left, nm_per_px):
        clipped = self._clip_polygon_vertical(pts_raw, x_line, keep_left=keep_left)
        if clipped.shape[0] < 3:
            return 0.0
        area_px2 = self.polygon_area(clipped)
        return float(area_px2) * float(nm_per_px) * float(nm_per_px)

    def export_split_results(self, max_dist=50.0, id_mode="event", use_display_id=True):
        self._export_pin_line_records()
        self._export_split_object_records(max_dist=max_dist, id_mode=id_mode, use_display_id=use_display_id)
        comparison_csv = self._export_frame_comparison()
        self._plot_frame_comparison(comparison_csv)

    def _export_pin_line_records(self):
        path = os.path.join(self.output_root, f"{self.gas_category}_pin_split_line.csv")
        fields = [
            "frame_id",
            "frame_name",
            "nm_per_px",
            "pin_status",
            "pin_cx_px",
            "pin_cy_px",
            "split_offset_px",
            "split_x_raw_px",
            "split_x_aligned_px",
            "split_x_aligned_nm",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rec in self.pin_line_records:
                row = dict(rec)
                row["split_offset_px"] = float(self.split_offset_px)
                writer.writerow(row)
        print(f" - {path}")

    def _export_split_object_records(self, max_dist=50.0, id_mode="event", use_display_id=True):
        export_ids = self._build_export_instance_ids(
            max_dist=max_dist,
            id_mode=id_mode,
            use_display_id=use_display_id,
        )
        if len(export_ids) != len(self.split_records):
            raise ValueError(
                f"Split record count mismatch: ids={len(export_ids)} split_records={len(self.split_records)}"
            )

        path = os.path.join(self.output_root, f"{self.gas_category}_pin_split_objects.csv")
        fields = [
            "instance_id",
            "frame_id",
            "frame_name",
            "side_by_centroid",
            "nm_per_px",
            "pin_status",
            "pin_cx_px",
            "pin_cy_px",
            "split_x_raw_px",
            "split_x_aligned_px",
            "split_x_aligned_nm",
            "cx_raw_px",
            "cy_raw_px",
            "cx_aligned_nm",
            "cy_aligned_nm",
            "signed_distance_to_line_px",
            "signed_distance_to_line_nm",
            "area_nm2",
            "clipped_left_area_nm2",
            "clipped_right_area_nm2",
            "crosses_split_line",
            "diameter_nm",
            "height_nm",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for instance_id, rec in zip(export_ids, self.split_records):
                row = {"instance_id": int(instance_id)}
                row.update(rec)
                writer.writerow(row)
        print(f" - {path}")

    def _export_frame_comparison(self):
        pin_by_frame = {int(r["frame_id"]): r for r in self.pin_line_records}
        frame_ids = sorted(pin_by_frame.keys())

        stats = defaultdict(lambda: {
            "left_count": 0,
            "right_count": 0,
            "unknown_count": 0,
            "left_area_nm2": 0.0,
            "right_area_nm2": 0.0,
            "unknown_area_nm2": 0.0,
            "left_clipped_area_nm2": 0.0,
            "right_clipped_area_nm2": 0.0,
            "crossing_count": 0,
        })

        for rec in self.split_records:
            fid = int(rec["frame_id"])
            side = str(rec["side_by_centroid"])
            area = float(rec["area_nm2"])
            if side == "left":
                stats[fid]["left_count"] += 1
                stats[fid]["left_area_nm2"] += area
            elif side == "right":
                stats[fid]["right_count"] += 1
                stats[fid]["right_area_nm2"] += area
            else:
                stats[fid]["unknown_count"] += 1
                stats[fid]["unknown_area_nm2"] += area

            stats[fid]["left_clipped_area_nm2"] += self._safe_float(rec["clipped_left_area_nm2"])
            stats[fid]["right_clipped_area_nm2"] += self._safe_float(rec["clipped_right_area_nm2"])
            stats[fid]["crossing_count"] += int(rec["crosses_split_line"])

        path = os.path.join(self.output_root, f"{self.gas_category}_pin_split_frame_comparison.csv")
        fields = [
            "frame_id",
            "frame_name",
            "nm_per_px",
            "pin_status",
            "pin_cx_px",
            "pin_cy_px",
            "split_offset_px",
            "split_x_raw_px",
            "split_x_aligned_px",
            "split_x_aligned_nm",
            "left_count",
            "right_count",
            "unknown_count",
            "left_area_nm2",
            "right_area_nm2",
            "unknown_area_nm2",
            "left_clipped_area_nm2",
            "right_clipped_area_nm2",
            "total_count",
            "total_area_nm2",
            "total_clipped_area_nm2",
            "left_minus_right_count",
            "left_minus_right_area_nm2",
            "left_minus_right_clipped_area_nm2",
            "left_count_fraction",
            "left_area_fraction",
            "left_clipped_area_fraction",
            "crossing_count",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for fid in frame_ids:
                pin = pin_by_frame[fid]
                st = stats[fid]
                total_count = st["left_count"] + st["right_count"] + st["unknown_count"]
                total_area = st["left_area_nm2"] + st["right_area_nm2"] + st["unknown_area_nm2"]
                total_clipped = st["left_clipped_area_nm2"] + st["right_clipped_area_nm2"]
                writer.writerow(
                    {
                        "frame_id": fid,
                        "frame_name": pin.get("frame_name", ""),
                        "nm_per_px": pin.get("nm_per_px", ""),
                        "pin_status": pin.get("pin_status", ""),
                        "pin_cx_px": pin.get("pin_cx_px", ""),
                        "pin_cy_px": pin.get("pin_cy_px", ""),
                        "split_offset_px": float(self.split_offset_px),
                        "split_x_raw_px": pin.get("split_x_raw_px", ""),
                        "split_x_aligned_px": pin.get("split_x_aligned_px", ""),
                        "split_x_aligned_nm": pin.get("split_x_aligned_nm", ""),
                        "left_count": st["left_count"],
                        "right_count": st["right_count"],
                        "unknown_count": st["unknown_count"],
                        "left_area_nm2": f"{st['left_area_nm2']:.6f}",
                        "right_area_nm2": f"{st['right_area_nm2']:.6f}",
                        "unknown_area_nm2": f"{st['unknown_area_nm2']:.6f}",
                        "left_clipped_area_nm2": f"{st['left_clipped_area_nm2']:.6f}",
                        "right_clipped_area_nm2": f"{st['right_clipped_area_nm2']:.6f}",
                        "total_count": total_count,
                        "total_area_nm2": f"{total_area:.6f}",
                        "total_clipped_area_nm2": f"{total_clipped:.6f}",
                        "left_minus_right_count": st["left_count"] - st["right_count"],
                        "left_minus_right_area_nm2": f"{st['left_area_nm2'] - st['right_area_nm2']:.6f}",
                        "left_minus_right_clipped_area_nm2": f"{st['left_clipped_area_nm2'] - st['right_clipped_area_nm2']:.6f}",
                        "left_count_fraction": self._fraction(st["left_count"], st["left_count"] + st["right_count"]),
                        "left_area_fraction": self._fraction(st["left_area_nm2"], st["left_area_nm2"] + st["right_area_nm2"]),
                        "left_clipped_area_fraction": self._fraction(st["left_clipped_area_nm2"], total_clipped),
                        "crossing_count": st["crossing_count"],
                    }
                )
        print(f" - {path}")
        return path

    @staticmethod
    def _safe_float(value):
        if value in ("", None):
            return 0.0
        return float(value)

    @staticmethod
    def _fraction(num, den):
        den = float(den)
        if den == 0.0:
            return ""
        return f"{float(num) / den:.6f}"

    def _plot_frame_comparison(self, comparison_csv):
        rows = []
        with open(comparison_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            print("No frame comparison rows to plot.")
            return

        frames = np.array([int(r["frame_id"]) for r in rows], dtype=np.int32)
        left_counts = np.array([float(r["left_count"]) for r in rows], dtype=np.float64)
        right_counts = np.array([float(r["right_count"]) for r in rows], dtype=np.float64)
        left_area = np.array([float(r["left_clipped_area_nm2"]) for r in rows], dtype=np.float64)
        right_area = np.array([float(r["right_clipped_area_nm2"]) for r in rows], dtype=np.float64)

        count_path = os.path.join(self.output_root, f"{self.gas_category}_pin_split_left_right_counts.png")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(frames, left_counts, label="left", color="#1f77b4", linewidth=1.6)
        ax.plot(frames, right_counts, label="right", color="#d62728", linewidth=1.6)
        ax.set_xlabel("Frame id")
        ax.set_ylabel("Instance count")
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.set_title(f"{self.gas_category}: left/right count split at {self._offset_display_label()}", loc="center")
        plt.tight_layout()
        plt.savefig(count_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved split count comparison plot: {count_path}")

        area_path = os.path.join(self.output_root, f"{self.gas_category}_pin_split_left_right_area.png")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(frames, left_area, label="left clipped area", color="#1f77b4", linewidth=1.6)
        ax.plot(frames, right_area, label="right clipped area", color="#d62728", linewidth=1.6)
        ax.set_xlabel("Frame id")
        ax.set_ylabel("Area (nm^2)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.set_title(f"{self.gas_category}: left/right clipped area split at {self._offset_display_label()}", loc="center")
        plt.tight_layout()
        plt.savefig(area_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved split area comparison plot: {area_path}")

        diff_path = os.path.join(self.output_root, f"{self.gas_category}_pin_split_left_minus_right.png")
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(frames, left_counts - right_counts, color="#2ca02c", linewidth=1.5)
        axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        axes[0].set_ylabel("Count diff")
        axes[0].grid(True, alpha=0.25)
        axes[1].plot(frames, left_area - right_area, color="#9467bd", linewidth=1.5)
        axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        axes[1].set_xlabel("Frame id")
        axes[1].set_ylabel("Area diff (nm^2)")
        axes[1].grid(True, alpha=0.25)
        fig.suptitle(f"{self.gas_category}: left minus right", y=0.98)
        plt.tight_layout()
        plt.savefig(diff_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved split difference plot: {diff_path}")

    def export_side_analyses(
        self,
        max_dist=50.0,
        min_track_length=0,
        frame_interval_s=1 / 30,
        bin_size_frames=1,
        evolution_step=2,
        debug_stats=True,
    ):
        original = self._capture_record_context()
        original_category = self.gas_category
        original_output_root = self.output_root

        try:
            for side in ("left", "right"):
                side_root = os.path.join(original_output_root, side)
                os.makedirs(side_root, exist_ok=True)
                self._apply_side_context(side, side_root, f"{original_category}_{side}")

                print(f"Running side analysis: {side}")
                self.export_results(max_dist=max_dist)
                self.plot_evolution(step=evolution_step)
                self.plot_centroid_trajectories(max_dist=max_dist)
                self.plot_area_trajectories(
                    max_dist=max_dist,
                    min_track_length=min_track_length,
                    debug_stats=debug_stats,
                )
                self.plot_frame_instance_count_and_total_area()
                self.plot_area_delta_vs_frame(per_frame=True, reducer="sum")
                self.plot_velocity_trajectories(
                    max_dist=max_dist,
                    min_track_length=min_track_length,
                    frame_interval_s=frame_interval_s,
                    bin_size_frames=bin_size_frames,
                    debug_stats=debug_stats,
                )
        finally:
            self._restore_record_context(original)
            self.gas_category = original_category
            self.output_root = original_output_root

    def _capture_record_context(self):
        return {
            "area_records": self.area_records,
            "contour_records": self.contour_records,
            "centroid_records": self.centroid_records,
            "object_records": self.object_records,
            "diameter_height_records": self.diameter_height_records,
        }

    def _restore_record_context(self, ctx):
        self.area_records = ctx["area_records"]
        self.contour_records = ctx["contour_records"]
        self.centroid_records = ctx["centroid_records"]
        self.object_records = ctx["object_records"]
        self.diameter_height_records = ctx["diameter_height_records"]

    def _apply_side_context(self, side, output_root, category_name):
        rec = self.side_records[side]
        self.area_records = rec["area_records"]
        self.contour_records = rec["contour_records"]
        self.centroid_records = rec["centroid_records"]
        self.object_records = rec["object_records"]
        self.diameter_height_records = rec["diameter_height_records"]
        self.output_root = output_root
        self.gas_category = category_name

    def annotate_split_line_on_rawframe(
        self,
        raw_frame_dir,
        output_dir=None,
        mask_alpha=115,
        frame_step=1,
        max_frames=None,
    ):
        frame_step = self._normalize_frame_step(frame_step)
        if output_dir is None:
            output_dir = os.path.join(self.output_root, "annotated_pin_split_rawframe")
        elif not os.path.isabs(output_dir):
            output_dir = os.path.join(self.output_root, output_dir)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Annotating split-line raw frames to {output_dir} (frame_step={frame_step})...")

        possible_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        left_rgb = (30, 150, 255)
        right_rgb = (230, 60, 150)
        pin_rgb = (240, 200, 0)
        line_rgb = (255, 255, 0)
        saved_count = 0

        json_files = self.json_files
        if max_frames is not None:
            json_files = json_files[: max(0, int(max_frames))]

        for frame_id, json_name in enumerate(json_files):
            if int(frame_id) % frame_step != 0:
                continue
            line_info = self._line_by_frame.get(int(frame_id))
            if not line_info or line_info.get("split_x_raw_px") in ("", None):
                continue

            frame_name = Path(json_name).stem
            raw_img_path = None
            for ext in possible_exts:
                candidate = os.path.join(raw_frame_dir, frame_name + ext)
                if os.path.exists(candidate):
                    raw_img_path = candidate
                    break
            if raw_img_path is None:
                continue

            try:
                with Image.open(raw_img_path) as raw_im:
                    bg = raw_im.convert("RGBA")
                    width, height = bg.size
                    font_px = max(18, min(32, int(round(min(width, height) * 0.028))))
                    try:
                        font = ImageFont.truetype("arial.ttf", font_px)
                    except OSError:
                        font = ImageFont.load_default()
                    stroke_w = int(max(1, round(font_px * 0.12)))

                    json_path = os.path.join(self.json_dir, json_name)
                    with open(json_path, "r", encoding="utf-8") as f:
                        jdata = json.load(f)

                    split_x = float(line_info["split_x_raw_px"])

                    mask_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                    mask_draw = ImageDraw.Draw(mask_layer)
                    gas_objects = []
                    pin_objects = []

                    for obj in jdata.get("objects", []):
                        cat = obj.get("category")
                        pts_raw = np.array(obj.get("segmentation", []), dtype=np.float32)
                        if pts_raw.shape[0] < 3:
                            continue
                        if cat == self.gas_category:
                            cx = float(np.mean(pts_raw[:, 0]))
                            side = "left" if cx < split_x else "right"
                            rgb = left_rgb if side == "left" else right_rgb
                            poly = [tuple(map(float, p)) for p in pts_raw]
                            mask_draw.polygon(poly, fill=rgb + (mask_alpha,), outline=None)
                            gas_objects.append((pts_raw, side, rgb))
                        elif cat == self.pin_category:
                            pin_objects.append(pts_raw)

                    img_out = Image.alpha_composite(bg, mask_layer).convert("RGB")
                    draw = ImageDraw.Draw(img_out)

                    for pts_raw, _side, rgb in gas_objects:
                        poly = [tuple(map(float, p)) for p in pts_raw]
                        draw.polygon(poly, outline=rgb, width=2)

                    for pts_raw in pin_objects:
                        poly = [tuple(map(float, p)) for p in pts_raw]
                        draw.polygon(poly, outline=pin_rgb, width=2)

                    line_w = max(3, int(round(min(width, height) * 0.002)))
                    draw.line([(split_x, 0), (split_x, height)], fill=line_rgb, width=line_w)

                    pin_cx = line_info.get("pin_cx_px")
                    pin_cy = line_info.get("pin_cy_px")
                    if pin_cx not in ("", None) and pin_cy not in ("", None):
                        r = max(5, int(round(font_px * 0.28)))
                        cx = float(pin_cx)
                        cy = float(pin_cy)
                        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=line_rgb, width=2)

                    label = self._offset_display_label()
                    label_x = min(max(split_x + 8, 4), max(4, width - 220))
                    draw.text(
                        (label_x, 8),
                        label,
                        fill=line_rgb,
                        font=font,
                        stroke_width=stroke_w,
                        stroke_fill="black",
                    )

                    out_path = os.path.join(output_dir, frame_name + ".png")
                    img_out.save(out_path)
                    saved_count += 1
            except Exception as e:
                print(f"Error annotating split raw frame {frame_name}: {e}")

        print(f"Split-line raw-frame annotation complete: {output_dir} ({saved_count} images saved)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze detections left/right of a vertical line at pin centroid plus a signed pixel offset."
    )
    parser.add_argument(
        "--json-dir",
        default=r"D:\code\zwl_NANO\outputs\zwl_roi_crops_blue_yellow_1024_tvl1_lam05_sharp\lable3100-7400_forward_fullsize",
    )
    parser.add_argument("--raw-frame-dir", default=r"D:\code\zwl_NANO\data\zwl")
    parser.add_argument(
        "--scale-csv",
        default=r"D:\code\zwl_NANO\outputs\zwl_scale_bar_detection\scalebar_for_analyze_rawframe.csv",
    )
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--gas-category", default="nanocluster")
    parser.add_argument("--pin-category", default="pin")
    parser.add_argument("--split-offset-px", type=float, default=-180.0)
    parser.add_argument("--scale-value-nm", type=float, default=20.0)
    parser.add_argument("--nm-per-px", type=float, default=None)
    parser.add_argument("--strict-scale-match", dest="strict_scale_match", action="store_true", default=True)
    parser.add_argument("--no-strict-scale-match", dest="strict_scale_match", action="store_false")
    parser.add_argument("--visualize-raw-frames", dest="visualize_raw_frames", action="store_true", default=True)
    parser.add_argument("--no-visualize-raw-frames", dest="visualize_raw_frames", action="store_false")
    parser.add_argument("--visualization-frame-step", type=int, default=100)
    parser.add_argument("--evolution-step", type=int, default=2)
    parser.add_argument("--max-dist", type=float, default=20.0)
    parser.add_argument("--min-track-length", type=int, default=0)
    parser.add_argument("--frame-interval-s", type=float, default=1 / 30)
    parser.add_argument("--bin-size-frames", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--skip-side-analyses", action="store_true")
    parser.add_argument("--skip-base-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    tracker = PinSplitGasTracker(
        json_dir=args.json_dir,
        image_path=args.image_path,
        scale_csv=args.scale_csv,
        scale_value_nm=args.scale_value_nm,
        nm_per_px=args.nm_per_px,
        strict_scale_match=args.strict_scale_match,
        gas_category=args.gas_category,
        pin_category=args.pin_category,
        split_offset_px=args.split_offset_px,
        output_root=args.output_root,
    )
    tracker.process_all_frames(max_frames=args.max_frames)
    tracker.export_results(max_dist=args.max_dist)
    tracker.export_split_results(max_dist=args.max_dist)

    if args.visualize_raw_frames:
        tracker.annotate_split_line_on_rawframe(
            raw_frame_dir=args.raw_frame_dir,
            output_dir="annotated_pin_split_rawframe",
            mask_alpha=120,
            frame_step=args.visualization_frame_step,
            max_frames=args.max_frames,
        )
    else:
        print("[skip] Raw-frame split-line visualization disabled.")

    if not args.skip_base_plots:
        tracker.plot_evolution(step=args.evolution_step)
        tracker.plot_centroid_trajectories(max_dist=args.max_dist)
        tracker.plot_area_trajectories(
            max_dist=args.max_dist,
            min_track_length=args.min_track_length,
            debug_stats=True,
        )
        tracker.plot_frame_instance_count_and_total_area()
        tracker.plot_area_delta_vs_frame(per_frame=True, reducer="sum")
        tracker.plot_velocity_trajectories(
            max_dist=args.max_dist,
            min_track_length=args.min_track_length,
            frame_interval_s=args.frame_interval_s,
            bin_size_frames=args.bin_size_frames,
            debug_stats=True,
        )

    if not args.skip_side_analyses:
        tracker.export_side_analyses(
            max_dist=args.max_dist,
            min_track_length=args.min_track_length,
            frame_interval_s=args.frame_interval_s,
            bin_size_frames=args.bin_size_frames,
            evolution_step=args.evolution_step,
            debug_stats=True,
        )


if __name__ == "__main__":
    main()
