"""Per-frame summary and evolution visualizations."""

from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class SummaryPlotMixin:
    """Provide per-frame summary and evolution visualizations."""

    def plot_area_delta_vs_frame(
        self,
        outname=None,
        out_csv=None,
        per_frame=True,
        reducer="sum",
    ):
        """Plot per-frame change in area (Δarea) as a single curve.

        This is computed from `self.area_records` by first aggregating all detections within
        the same frame (default: sum), then taking a first-order difference between
        consecutive frames.

        Args:
            outname: Output PNG name.
            out_csv: Optional CSV output for the delta series.
            per_frame: If True, normalize by delta_frame (handles skipped frames).
            reducer: How to aggregate multiple objects in the same frame: 'sum' or 'mean'.
        """
        if len(self.area_records) == 0:
            print("No area records to plot area delta.")
            return

        from collections import defaultdict

        reducer_key = str(reducer).strip().lower()
        if reducer_key not in {"sum", "mean"}:
            raise ValueError(f"reducer must be 'sum' or 'mean', got {reducer}")

        areas_by_frame = defaultdict(list)  # frame_id -> list[area_nm2]
        name_by_frame = {}
        for frame_id, frame_name, _nm_per_px, area_nm2 in self.area_records:
            fid = int(frame_id)
            areas_by_frame[fid].append(float(area_nm2))
            if fid not in name_by_frame:
                name_by_frame[fid] = str(frame_name)

        frame_ids = sorted(areas_by_frame.keys())
        if len(frame_ids) < 2:
            print("Not enough frames to compute area delta (need >= 2).")
            return

        area_series = []  # (frame_id, frame_name, area_agg_nm2)
        for fid in frame_ids:
            vals = areas_by_frame[fid]
            if len(vals) == 0:
                continue
            if reducer_key == "mean":
                a = float(np.mean(vals))
            else:
                a = float(np.sum(vals))
            area_series.append((int(fid), name_by_frame.get(int(fid), str(fid)), a))

        # ensure sorted
        area_series.sort(key=lambda t: int(t[0]))

        # delta aligned to current frame
        delta_points = []  # (frame_id, frame_name, delta_area_nm2_per_frame)
        prev_f, _prev_name, prev_a = area_series[0]
        for cur_f, cur_name, cur_a in area_series[1:]:
            df = int(cur_f) - int(prev_f)
            if df <= 0:
                prev_f, prev_a = cur_f, cur_a
                continue
            da = float(cur_a) - float(prev_a)
            if bool(per_frame):
                da = da / float(df)
            delta_points.append((int(cur_f), str(cur_name), float(da)))
            prev_f, prev_a = cur_f, cur_a

        if len(delta_points) == 0:
            print("Area delta series is empty after processing.")
            return

        if outname is None:
            outname = os.path.join(
                self.output_root, f"{self.target_category}_area_delta_vs_frame.png"
            )
        elif not os.path.isabs(outname):
            outname = os.path.join(self.output_root, outname)

        if out_csv is None:
            out_csv = os.path.join(
                self.output_root, f"{self.target_category}_area_delta_vs_frame.csv"
            )
        elif not os.path.isabs(out_csv):
            out_csv = os.path.join(self.output_root, out_csv)

        frames = np.array([p[0] for p in delta_points], dtype=np.int32)
        deltas = np.array([p[2] for p in delta_points], dtype=np.float64)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(frames, deltas, color="#1f77b4", linewidth=1.6)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.set_xlabel("Frame id")
        ylab = "ΔArea (nm^2/frame)" if bool(per_frame) else "ΔArea (nm^2)"
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)

        agg_label = "sum" if reducer_key == "sum" else "mean"
        ax.set_title(
            f"{self.target_category}: per-frame area change (Δarea), frame-agg={agg_label}",
            loc="center",
        )

        plt.tight_layout()
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        print(f"Saved area delta plot: {outname}")

        # export delta CSV
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "frame_id",
                    "frame_name",
                    "delta_area_nm2_per_frame" if bool(per_frame) else "delta_area_nm2",
                ]
            )
            for fid, fname, da in delta_points:
                writer.writerow([int(fid), str(fname), f"{float(da):.6f}"])
        print(f" - {out_csv}")

    def plot_frame_instance_count_and_total_area(self, outname=None, out_csv=None):
        """Plot per-frame instance count and total area as two separate figures.

        Uses `self.area_records` where each row is one detected instance in a frame.
        - instance_count(frame): number of instances in this frame
        - total_area_nm2(frame): sum of all instance areas in this frame
        """
        if len(self.area_records) == 0:
            print("No area records to plot frame totals.")
            return

        from collections import defaultdict

        count_by_frame = defaultdict(int)
        area_sum_by_frame = defaultdict(float)
        name_by_frame = {}

        for frame_id, frame_name, _nm_per_px, area_nm2 in self.area_records:
            fid = int(frame_id)
            count_by_frame[fid] += 1
            area_sum_by_frame[fid] += float(area_nm2)
            if fid not in name_by_frame:
                name_by_frame[fid] = str(frame_name)

        frame_ids = sorted(count_by_frame.keys())
        if len(frame_ids) == 0:
            print("No valid frame statistics to plot.")
            return

        if outname is None:
            count_plot_path = os.path.join(
                self.output_root, f"{self.target_category}_frame_instance_count.png"
            )
            area_plot_path = os.path.join(
                self.output_root, f"{self.target_category}_frame_total_area.png"
            )
        else:
            # If outname is provided, treat it as a shared prefix for two plot files.
            if not os.path.isabs(outname):
                outname = os.path.join(self.output_root, outname)
            base, ext = os.path.splitext(outname)
            if ext == "":
                ext = ".png"
            count_plot_path = f"{base}_instance_count{ext}"
            area_plot_path = f"{base}_total_area{ext}"

        if out_csv is None:
            out_csv = os.path.join(
                self.output_root, f"{self.target_category}_frame_count_area.csv"
            )
        elif not os.path.isabs(out_csv):
            out_csv = os.path.join(self.output_root, out_csv)

        frames = np.array(frame_ids, dtype=np.int32)
        counts = np.array([count_by_frame[fid] for fid in frame_ids], dtype=np.int32)
        areas = np.array(
            [area_sum_by_frame[fid] for fid in frame_ids], dtype=np.float64
        )

        # Plot 1: instance count only
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(frames, counts, color="#1f77b4", linewidth=1.8)
        ax.set_xlabel("Frame id")
        ax.set_ylabel("Instance count")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{self.target_category}: per-frame instance count", loc="center")
        plt.tight_layout()
        plt.savefig(count_plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved frame instance-count plot: {count_plot_path}")
        plt.close(fig)

        # Plot 2: total area only
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(frames, areas, color="#d62728", linewidth=1.8)
        ax.set_xlabel("Frame id")
        ax.set_ylabel("Total area (nm^2)")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{self.target_category}: per-frame total area", loc="center")
        plt.tight_layout()
        plt.savefig(area_plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved frame total-area plot: {area_plot_path}")
        plt.close(fig)

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["frame_id", "frame_name", "instance_count", "total_area_nm2"]
            )
            for fid in frame_ids:
                writer.writerow(
                    [
                        int(fid),
                        name_by_frame.get(int(fid), str(fid)),
                        int(count_by_frame[fid]),
                        f"{float(area_sum_by_frame[fid]):.6f}",
                    ]
                )
        print(f" - {out_csv}")

    def plot_evolution(self, step=200):
        fig, ax = plt.subplots(figsize=(8, 8))
        point_arrays = []

        cmap = plt.cm.plasma
        norm = Normalize(vmin=0, vmax=len(self.json_files) - 1)

        for row in self.contour_records:
            frame_id = row[0]
            if frame_id % step != 0:
                continue

            pts = []
            # row format: [frame_id, frame_name, "(x_nm,y_nm)", ...]
            for item in row[2:]:
                x, y = map(float, item.strip("()").split(","))
                pts.append([x, y])

            pts = np.array(pts)
            pts = np.vstack([pts, pts[0]])
            point_arrays.append(pts)

            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=cmap(norm(frame_id)),
                linewidth=1.5,
                alpha=0.85,
            )

        if not self._set_nm_axes(ax, point_arrays):
            plt.close(fig)
            print("No contour records to plot.")
            return

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Frame id colorbar: same height as the axes
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.10)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Frame id")

        ax.set_title(
            f"{self.target_category} domain evolution (pin-referenced)", loc="center"
        )
        plt.tight_layout()
        # add a visible border around the axes
        from matplotlib.patches import Rectangle

        border_width = 3
        border_color = "black"
        rect = Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            edgecolor=border_color,
            linewidth=border_width,
            zorder=10,
            clip_on=False,
        )
        ax.add_patch(rect)
        outname = os.path.join(
            self.output_root, f"{self.target_category}_evolution.png"
        )
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved evolution plot: {outname}")
