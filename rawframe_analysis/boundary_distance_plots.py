"""Tracked pair boundary-distance visualizations across frames."""

from __future__ import annotations

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


class BoundaryDistancePlotMixin:
    """Plot one distance trajectory for each tracked object pair."""

    def plot_boundary_distances_vs_frame(
        self,
        max_dist=50.0,
        min_pair_length=1,
        max_plot_pairs=500,
        max_legend_items=60,
        debug_stats=False,
    ):
        """Track both categories and plot every retained pair as its own line.

        Particle and droplet identities are linked independently between
        consecutive frames using centroid distance. Lines are broken whenever a
        pair has no observation in an intermediate frame.
        """
        if not self.compute_boundary_distances_enabled:
            print("Boundary-distance analysis is disabled; no distance plots saved.")
            return []

        particle_pairs, particle_droplet_pairs = self._tracked_boundary_distance_series(
            max_dist=max_dist
        )
        minimum_length = max(1, int(min_pair_length))
        particle_pairs = {
            pair: points
            for pair, points in particle_pairs.items()
            if len(points) >= minimum_length
        }
        particle_droplet_pairs = {
            pair: points
            for pair, points in particle_droplet_pairs.items()
            if len(points) >= minimum_length
        }

        if debug_stats:
            print(
                f"[debug] boundary-distance pairs: particle-particle={len(particle_pairs)}, "
                f"particle-droplet={len(particle_droplet_pairs)}, "
                f"max_dist_nm={float(max_dist):.6f}, min_pair_length={minimum_length}"
            )

        outputs = []
        particle_pair_path = self._plot_pair_distance_series(
            particle_pairs,
            filename=(
                f"{self.particle_category}_to_{self.particle_category}"
                "_boundary_distance_vs_frame.png"
            ),
            title=(
                f"{self.particle_category}-{self.particle_category} pair "
                "boundary distances"
            ),
            label_builder=lambda pair: f"P{pair[0]}-P{pair[1]}",
            max_plot_pairs=max_plot_pairs,
            max_legend_items=max_legend_items,
        )
        if particle_pair_path is not None:
            outputs.append(particle_pair_path)

        particle_droplet_path = self._plot_pair_distance_series(
            particle_droplet_pairs,
            filename=(
                f"{self.particle_category}_to_{self.droplet_category}"
                "_boundary_distance_vs_frame.png"
            ),
            title=(
                f"{self.particle_category}-{self.droplet_category} pair "
                "boundary distances"
            ),
            label_builder=lambda pair: f"P{pair[0]}-D{pair[1]}",
            max_plot_pairs=max_plot_pairs,
            max_legend_items=max_legend_items,
        )
        if particle_droplet_path is not None:
            outputs.append(particle_droplet_path)
        return outputs

    def _tracked_boundary_distance_series(self, max_dist):
        particle_ids = self._tracked_category_ids(
            self.boundary_particle_records,
            max_dist=max_dist,
            category=self.particle_category,
        )
        droplet_ids = self._tracked_category_ids(
            self.boundary_droplet_records,
            max_dist=max_dist,
            category=self.droplet_category,
        )

        particle_pairs = defaultdict(list)
        for row in self.particle_particle_distance_records:
            frame_id = int(row[0])
            ids = particle_ids.get(frame_id, [])
            first_index = int(row[3]) - 1
            second_index = int(row[5]) - 1
            if first_index >= len(ids) or second_index >= len(ids):
                continue
            pair = tuple(sorted((int(ids[first_index]), int(ids[second_index]))))
            particle_pairs[pair].append((frame_id, float(row[7])))

        particle_droplet_pairs = defaultdict(list)
        for row in self.particle_droplet_distance_records:
            frame_id = int(row[0])
            frame_particle_ids = particle_ids.get(frame_id, [])
            frame_droplet_ids = droplet_ids.get(frame_id, [])
            particle_index = int(row[3]) - 1
            droplet_index = int(row[5]) - 1
            if particle_index >= len(frame_particle_ids) or droplet_index >= len(
                frame_droplet_ids
            ):
                continue
            pair = (
                int(frame_particle_ids[particle_index]),
                int(frame_droplet_ids[droplet_index]),
            )
            particle_droplet_pairs[pair].append((frame_id, float(row[7])))

        return dict(particle_pairs), dict(particle_droplet_pairs)

    def _plot_pair_distance_series(
        self,
        series_by_pair,
        *,
        filename,
        title,
        label_builder,
        max_plot_pairs,
        max_legend_items,
    ):
        if not series_by_pair:
            print(f"No tracked pair records available for {title}.")
            return None

        pairs = sorted(
            series_by_pair,
            key=lambda pair: (
                -len(series_by_pair[pair]),
                min(point[0] for point in series_by_pair[pair]),
                pair,
            ),
        )
        plot_limit = self._positive_plot_limit(max_plot_pairs)
        if plot_limit is not None and len(pairs) > plot_limit:
            print(
                f"[warn] {title} has {len(pairs)} pair trajectories; "
                f"drawing the longest {plot_limit}."
            )
            pairs = pairs[:plot_limit]

        fig, ax = plt.subplots(figsize=(13, 6.5))
        color_map = plt.get_cmap("tab20")
        handles = []
        labels = []
        line_styles = ("-", "--", "-.", ":")
        markers = ("o", "s", "^", "D", "v")
        for pair_index, pair in enumerate(pairs):
            frames, distances = self._line_with_frame_gaps(series_by_pair[pair])
            (line,) = ax.plot(
                frames,
                distances,
                color=color_map(pair_index % 20),
                linestyle=line_styles[(pair_index // 20) % len(line_styles)],
                linewidth=1.45,
                marker=markers[(pair_index // 20) % len(markers)],
                markersize=2.8,
                alpha=0.88,
            )
            handles.append(line)
            labels.append(label_builder(pair))

        self._style_distance_axis(ax, title)
        legend_limit = self._positive_plot_limit(max_legend_items)
        if legend_limit is None or len(handles) <= legend_limit:
            columns = 1 if len(handles) <= 15 else 2
            fig.subplots_adjust(right=0.78 if columns == 1 else 0.70)
            ax.legend(
                handles,
                labels,
                title="Tracked pair ID",
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                frameon=False,
                ncol=columns,
                fontsize=11,
                title_fontsize=12,
            )
        else:
            # The user needs every pair to remain identifiable. For a large
            # number of pairs, move a compact multi-column legend below the
            # chart instead of omitting it.
            columns = max(4, min(9, int(np.ceil(len(handles) / 12.0))))
            fig.set_size_inches(17, 11, forward=True)
            fig.subplots_adjust(bottom=0.34)
            ax.legend(
                handles,
                labels,
                title="Tracked pair ID",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.17),
                frameon=False,
                ncol=columns,
                fontsize=9,
                title_fontsize=11,
                columnspacing=1.0,
                handlelength=2.4,
            )

        output_path = os.path.join(self.output_root, filename)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved tracked pair boundary-distance plot: {output_path}")
        return output_path

    @staticmethod
    def _line_with_frame_gaps(points):
        ordered = sorted(points, key=lambda point: int(point[0]))
        frames = []
        distances = []
        previous_frame = None
        for frame, distance in ordered:
            frame = int(frame)
            if previous_frame is not None and frame > previous_frame + 1:
                frames.append(np.nan)
                distances.append(np.nan)
            frames.append(float(frame))
            distances.append(float(distance))
            previous_frame = frame
        return (
            np.asarray(frames, dtype=np.float64),
            np.asarray(distances, dtype=np.float64),
        )

    @staticmethod
    def _style_distance_axis(ax, title):
        ax.set_xlabel("Frame id")
        ax.set_ylabel("Boundary distance (nm)")
        ax.set_title(title, loc="center")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
