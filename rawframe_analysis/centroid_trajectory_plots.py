"""Centroid trajectory visualization."""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class CentroidTrajectoryPlotMixin:
    """Provide centroid trajectory visualization."""

    def plot_centroid_trajectories(self, max_dist=50.0, max_fastplot_points=100000):
        """
        Build simple greedy tracks by linking centroids in consecutive frames
        when their distance is <= max_dist. Save plot to PNG.
        NOTE: max_dist is in nm because centroids are stored in nm.
        """
        if len(self.centroid_records) == 0:
            print("No centroid records to plot.")
            return

        fastplot_enabled = bool(getattr(self, "fastplot_enabled", True))
        if fastplot_enabled:
            rows = self.centroid_records
            total_points = len(rows)
            if total_points == 0:
                print("No centroid records to plot.")
                return

            max_points = (
                int(max_fastplot_points) if max_fastplot_points is not None else 0
            )
            if max_points > 0 and total_points > max_points:
                stride = int(np.ceil(float(total_points) / float(max_points)))
                rows = rows[::stride]
                print(
                    f"[warn] {self.target_category} centroid plot has {total_points} points; "
                    f"drawing every {stride}th point ({len(rows)} points)."
                )

            frames = np.array([int(r[0]) for r in rows], dtype=np.int32)
            xs = np.array([float(r[3]) for r in rows], dtype=np.float64)
            ys = np.array([float(r[4]) for r in rows], dtype=np.float64)
            pts_for_axes = np.column_stack([xs, ys])

            fig, ax = plt.subplots(figsize=(8, 8))
            cmap = plt.cm.plasma
            norm = Normalize(vmin=0, vmax=max(len(self.json_files) - 1, 1))
            ax.scatter(
                xs, ys, c=frames, cmap=cmap, norm=norm, s=1, linewidths=0, alpha=0.9
            )

            if not self._set_nm_axes(ax, [pts_for_axes]):
                plt.close(fig)
                print("No centroid records to plot.")
                return

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.10)
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label("Frame id")

            ax.set_title(
                f"{self.target_category} centroid positions (fastplot, time-colored) | points={len(rows)}",
                loc="center",
            )
            plt.tight_layout()
            outname = os.path.join(
                self.output_root, f"{self.target_category}_centroid_trajectories.png"
            )
            from matplotlib.patches import Rectangle

            rect = Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                fill=False,
                edgecolor="black",
                linewidth=3,
                zorder=10,
                clip_on=False,
            )
            ax.add_patch(rect)
            plt.savefig(outname, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved centroid trajectories plot: {outname}")
            return

        from collections import defaultdict

        # organize centroids by frame
        by_frame = defaultdict(list)
        for frame_id, frame_name, nm_per_px, cx_nm, cy_nm in self.centroid_records:
            by_frame[int(frame_id)].append((frame_name, float(cx_nm), float(cy_nm)))

        tracks = []  # each track: {'last_frame': int, 'points': [(frame,cx,cy), ...]}

        for frame in sorted(by_frame.keys()):
            pts = by_frame[frame]
            assigned = [False] * len(pts)

            # try to extend existing tracks from previous frame
            for t in tracks:
                if t["last_frame"] != frame - 1:
                    continue
                last_x, last_y = t["points"][-1][2], t["points"][-1][3]
                best_idx = None
                best_dist = float("inf")
                for i, (frame_name, cx_nm, cy_nm) in enumerate(pts):
                    if assigned[i]:
                        continue
                    d = np.hypot(cx_nm - last_x, cy_nm - last_y)
                    if d < best_dist:
                        best_dist = d
                        best_idx = i

                if best_idx is not None and best_dist <= max_dist:
                    frame_name, cx_nm, cy_nm = pts[best_idx]
                    t["points"].append((frame, frame_name, cx_nm, cy_nm))
                    t["last_frame"] = frame
                    assigned[best_idx] = True

            # create new tracks for unassigned centroids
            for i, (frame_name, cx_nm, cy_nm) in enumerate(pts):
                if not assigned[i]:
                    tracks.append(
                        {
                            "last_frame": frame,
                            "points": [(frame, frame_name, cx_nm, cy_nm)],
                        }
                    )

        # plotting
        fig, ax = plt.subplots(figsize=(8, 8))
        point_arrays = []

        # color by frame (time axis) — use same colormap/norm as evolution
        cmap = plt.cm.plasma
        norm = Normalize(vmin=0, vmax=len(self.json_files) - 1)

        for idx, t in enumerate(tracks):

            frames = np.array([p[0] for p in t["points"]])
            pts = np.array([[p[2], p[3]] for p in t["points"]])
            if pts.shape[0] == 0:
                continue
            point_arrays.append(pts)

            # draw colored segments between consecutive points according to the earlier frame
            for i in range(len(pts) - 1):
                col = cmap(norm(frames[i]))
                ax.plot(
                    pts[i : i + 2, 0],
                    pts[i : i + 2, 1],
                    "-",
                    color=col,
                    linewidth=1,
                    alpha=0.95,
                )

            # scatter points colored by their frame
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=frames, cmap=cmap, norm=norm, s=1)

        if not self._set_nm_axes(ax, point_arrays):
            plt.close(fig)
            print("No centroid records to plot.")
            return

        # add colorbar (time axis)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Frame id colorbar: same height as the axes
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.10)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Frame id")

        ax.set_title(
            f"{self.target_category} centroid trajectories (time-colored)", loc="center"
        )
        plt.tight_layout()
        outname = os.path.join(
            self.output_root, f"{self.target_category}_centroid_trajectories.png"
        )
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
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved centroid trajectories plot: {outname}")
