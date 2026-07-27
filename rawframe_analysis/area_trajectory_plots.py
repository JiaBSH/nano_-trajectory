"""Tracked-object area trajectory visualization."""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt


class AreaTrajectoryPlotMixin:
    """Provide tracked-object area trajectory visualization."""

    def plot_area_trajectories(
        self,
        max_dist=50.0,
        min_track_length=1,
        outname=None,
        id_mode="event",
        debug_stats=False,
        max_plot_tracks=500,
        max_legend_items=60,
        annotate_ids_max=80,
    ):
        """Plot each droplet's area-vs-frame curve in one figure.

        Tracks are built by greedy centroid linking.
        NOTE: max_dist is in nm because centroids are stored in nm.
        """
        if len(self.object_records) == 0:
            print("No object records to plot area trajectories.")
            return

        by_frame = self._object_detections_by_frame()

        if bool(debug_stats):
            n_frames = len(by_frame)
            n_dets = sum(len(v) for v in by_frame.values())
            print(
                f"[debug] {self.target_category} area: frames_with_detections={n_frames}, total_detections={n_dets}, "
                f"id_mode={id_mode}, max_dist_nm={float(max_dist)}, min_track_length={int(min_track_length)}"
            )

        if str(id_mode).lower() == "greedy":
            tracks = self._build_greedy_tracks(by_frame, max_dist=max_dist)
            if bool(debug_stats):
                print(
                    f"[debug] {self.target_category} area: greedy tracks before length filter={len(tracks)}"
                )
            tracks = [t for t in tracks if len(t["points"]) >= int(min_track_length)]
            if bool(debug_stats):
                print(
                    f"[debug] {self.target_category} area: greedy tracks after length filter={len(tracks)}"
                )
            series_by_id = {
                int(track_id): [
                    (p[0], p[1], p[2], p[3], p[4], p[5]) for p in t["points"]
                ]
                for track_id, t in enumerate(tracks)
            }
        else:
            if by_frame is self._object_detections_by_frame():
                series_by_id, _assigned_ids_by_frame, _events = (
                    self._event_id_series_for_object_records(max_dist=max_dist)
                )
            else:
                series_by_id, _events = self._build_event_id_series(
                    by_frame, max_dist=max_dist
                )
            if bool(debug_stats):
                print(
                    f"[debug] {self.target_category} area: event ids before length filter={len(series_by_id)}"
                )
            series_by_id = {
                k: v for k, v in series_by_id.items() if len(v) >= int(min_track_length)
            }
            if bool(debug_stats):
                kept = len(series_by_id)
                max_iid = max(series_by_id.keys()) if kept > 0 else None
                print(
                    f"[debug] {self.target_category} area: event ids after length filter={kept}, max_instance_id={max_iid}"
                )

        if outname is None:
            outname = os.path.join(
                self.output_root, f"{self.target_category}_area_trajectories.png"
            )
        elif not os.path.isabs(outname):
            outname = os.path.join(self.output_root, outname)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlabel("Frame id")
        ax.set_ylabel("Area (nm^2)")
        ax.grid(True, alpha=0.25)

        # color cycle (good for dozens of tracks; for hundreds, they'll repeat)
        cmap = plt.cm.tab20

        fastplot_enabled = bool(getattr(self, "fastplot_enabled", True))
        plot_max_tracks = (
            self._positive_plot_limit(max_plot_tracks) if fastplot_enabled else None
        )
        legend_max_items = (
            self._positive_plot_limit(max_legend_items) if fastplot_enabled else None
        )
        annotate_max = (
            self._positive_plot_limit(annotate_ids_max) if fastplot_enabled else None
        )

        display_id_of = self._display_id_mapping(series_by_id)
        instance_ids, total_instance_ids = self._select_plot_instance_ids(
            series_by_id,
            max_plot_tracks=plot_max_tracks,
        )
        if total_instance_ids > len(instance_ids):
            print(
                f"[warn] {self.target_category} area plot has {total_instance_ids} tracks; "
                f"drawing the longest {len(instance_ids)} tracks only. Full series is still exported."
            )

        if bool(debug_stats):
            max_disp = max(display_id_of.values()) if len(display_id_of) > 0 else 0
            print(
                f"[debug] {self.target_category} area: plotted_ids={len(instance_ids)}, "
                f"total_ids={total_instance_ids}, display_id_max={max_disp}"
            )

        line_handles = []
        line_labels = []

        skipped_empty = []

        for instance_id in instance_ids:
            pts = series_by_id[instance_id]
            frames = np.array([p[0] for p in pts], dtype=np.int32)
            areas = np.array([p[5] for p in pts], dtype=np.float32)
            if frames.size == 0:
                if bool(debug_stats):
                    skipped_empty.append(int(instance_id))
                continue
            order = np.argsort(frames)
            frames = frames[order]
            areas = areas[order]

            disp_id = display_id_of.get(int(instance_id), 0)
            color = cmap(int(disp_id) % 20)
            (line,) = ax.plot(frames, areas, color=color, linewidth=1.2, alpha=0.85)
            line_handles.append(line)
            line_id_label = str(int(disp_id) if disp_id > 0 else int(instance_id))
            line_labels.append(line_id_label)

            if annotate_max is None or len(instance_ids) <= int(annotate_max):
                try:
                    x0 = float(frames[0])
                    y0 = float(areas[0])
                    ax.annotate(
                        line_id_label,
                        xy=(x0, y0),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=12,
                        color=color,
                        bbox={
                            "boxstyle": "round,pad=0.2",
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.65,
                        },
                    )
                except Exception:
                    pass

        if bool(debug_stats):
            if len(skipped_empty) > 0:
                print(
                    f"[debug] {self.target_category} area: skipped_empty_ids={len(skipped_empty)} head={skipped_empty[:20]}"
                )
            print(
                f"[debug] {self.target_category} area: drawn_lines={len(line_handles)} legend_items={len(line_labels)}"
            )

        # 图例：自适应布局，避免挡线 & 避免图被挤得很“扁”
        leg = None
        if len(line_handles) > 0:
            n_items = len(line_handles)
            if legend_max_items is not None and n_items > int(legend_max_items):
                print(
                    f"[warn] Skip area legend: {n_items} plotted tracks exceed "
                    f"max_legend_items={int(legend_max_items)}."
                )
            elif n_items <= 20:
                fig.set_size_inches(12, 6, forward=True)
                fig.subplots_adjust(right=0.80)
                leg = ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=14,
                    ncol=1,
                )
            elif n_items <= 60:
                fig.set_size_inches(14, 6, forward=True)
                fig.subplots_adjust(right=0.78)
                leg = ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=13,
                    ncol=2,
                    columnspacing=0.8,
                    handlelength=1.2,
                )
            else:
                rows_target = 12
                ncol = int(np.ceil(float(n_items) / float(rows_target)))
                ncol = max(4, min(10, ncol))
                fig.set_size_inches(16, 11.0, forward=True)
                fig.subplots_adjust(bottom=0.36)
                leg = ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.22),
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=12,
                    ncol=ncol,
                    columnspacing=0.8,
                    handlelength=1.2,
                )

        ax.set_title(
            f"{self.target_category}: area vs frame | plotted={len(instance_ids)} of {total_instance_ids} tracks",
            loc="center",
        )
        plt.tight_layout()
        plt.savefig(
            outname,
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=((leg,) if leg is not None else None),
        )
        plt.close(fig)
        print(f"Saved area trajectories plot: {outname}")

        # also export tracked series for downstream analysis
        if str(id_mode).lower() == "greedy":
            self.export_tracked_area_results(tracks)
        else:
            self.export_id_series(series_by_id)
