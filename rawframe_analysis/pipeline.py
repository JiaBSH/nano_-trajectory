"""Application orchestration for raw-frame analysis.

The pipeline depends on a tracker interface and configuration abstractions rather
than constructing global state. This keeps CLI parsing, configuration persistence,
and scientific calculations independent from one another.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .config import ConfigSnapshot, ConfigSnapshotWriter, RawFrameConfig


class TrackerProtocol(Protocol):
    def process_all_frames(self) -> Any: ...

    def export_results(self, **kwargs: Any) -> Any: ...

    def validate_exported_csv_ids(self) -> Any: ...

    def annotate_images_on_rawframe(self, **kwargs: Any) -> Any: ...

    def annotate_allcategories_on_rawframe(self, **kwargs: Any) -> Any: ...

    def annotate_boundary_pair_ids_on_rawframe(self, **kwargs: Any) -> Any: ...

    def plot_evolution(self, **kwargs: Any) -> Any: ...

    def plot_centroid_trajectories(self, **kwargs: Any) -> Any: ...

    def plot_area_trajectories(self, **kwargs: Any) -> Any: ...

    def plot_frame_instance_count_and_total_area(self, **kwargs: Any) -> Any: ...

    def plot_area_delta_vs_frame(self, **kwargs: Any) -> Any: ...

    def plot_velocity_trajectories(self, **kwargs: Any) -> Any: ...

    def plot_boundary_distances_vs_frame(self, **kwargs: Any) -> Any: ...


TrackerFactory = Callable[..., TrackerProtocol]
LogFunction = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class PipelineStep:
    name: str
    action: Callable[[], Any]


@dataclass(frozen=True, slots=True)
class RunResult:
    tracker: TrackerProtocol
    config_snapshot: ConfigSnapshot
    completed_steps: tuple[str, ...]


class AnalysisPipeline:
    """Run configured analysis steps without owning their implementations."""

    def __init__(
        self,
        config: RawFrameConfig,
        tracker_factory: TrackerFactory,
        *,
        snapshot_writer: ConfigSnapshotWriter | None = None,
        log: LogFunction = print,
    ) -> None:
        self._config = config
        self._tracker_factory = tracker_factory
        self._snapshot_writer = snapshot_writer or ConfigSnapshotWriter()
        self._log = log

    def run(self) -> RunResult:
        self._config.validate()
        self._config.validate_input_paths()
        if not self._config.output.root:
            raise ValueError("Output root must be resolved before running the pipeline")

        Path(self._config.output.root).mkdir(parents=True, exist_ok=True)
        snapshot = self._snapshot_writer.write(self._config)
        self._log(f"[config] effective config: {snapshot.latest}")
        self._log(f"[config] archived snapshot: {snapshot.archived}")

        tracker = self._create_tracker()
        completed: list[str] = []
        for step in self._build_steps(tracker):
            self._timed_step(step)
            completed.append(step.name)

        if not (
            self._config.annotations.save_target_raw_frames
            or self._config.annotations.save_all_category_raw_frames
            or self._config.annotations.save_boundary_pair_id_raw_frames
        ):
            self._log("[skip] Raw-frame visualization disabled.")

        return RunResult(
            tracker=tracker,
            config_snapshot=snapshot,
            completed_steps=tuple(completed),
        )

    def _create_tracker(self) -> TrackerProtocol:
        cfg = self._config
        return self._tracker_factory(
            json_dir=cfg.input.json_dir,
            image_path=cfg.input.image_path,
            scale_csv=cfg.input.scale_csv,
            scale_value_nm=cfg.input.scale_value_nm,
            nm_per_px=cfg.input.nm_per_px,
            strict_scale_match=cfg.input.strict_scale_match,
            target_category=cfg.analysis.target_category,
            pin_category=cfg.analysis.pin_category,
            pin_reference_enabled=cfg.analysis.pin_reference_enabled,
            skip_frames_without_pin=cfg.analysis.skip_frames_without_pin,
            pin_centroid_smoothing_alpha=cfg.analysis.pin_centroid_smoothing_alpha,
            max_particle_pin_distance_nm=cfg.analysis.max_particle_pin_distance_nm,
            fastplot_enabled=cfg.analysis.fastplot_enabled,
            compute_diameter_height_enabled=cfg.analysis.compute_diameter_height_enabled,
            compute_boundary_distances_enabled=cfg.analysis.compute_boundary_distances_enabled,
            particle_category=cfg.analysis.particle_category,
            droplet_category=cfg.analysis.droplet_category,
            instance_overlap_postprocess_enabled=cfg.analysis.instance_overlap_postprocess_enabled,
            same_category_containment_threshold=cfg.analysis.same_category_containment_threshold,
            output_root=cfg.output.root,
        )

    def _build_steps(self, tracker: TrackerProtocol) -> list[PipelineStep]:
        cfg = self._config
        plots = cfg.plots
        annotations = cfg.annotations
        tracking_max_dist = cfg.output.export_max_dist_nm
        steps = [PipelineStep("process_all_frames", tracker.process_all_frames)]

        if cfg.output.export_csv_results:
            steps.append(
                PipelineStep(
                    "export_results",
                    lambda: tracker.export_results(
                        max_dist=cfg.output.export_max_dist_nm,
                        id_mode="event",
                        use_display_id=cfg.output.export_use_display_id,
                    ),
                )
            )

        if annotations.save_target_raw_frames:
            target_output_dir = annotations.target_output_dir or (
                f"annotated_{cfg.analysis.target_category}_rawframe"
            )
            steps.append(
                PipelineStep(
                    "annotate_images_on_rawframe",
                    lambda: tracker.annotate_images_on_rawframe(
                        raw_frame_dir=cfg.input.raw_frame_dir,
                        output_dir=target_output_dir,
                        label_ids=annotations.target_label_ids,
                        max_dist=tracking_max_dist,
                        mask_alpha=annotations.mask_alpha,
                        frame_step=annotations.frame_step,
                    ),
                )
            )

        if annotations.save_all_category_raw_frames:
            steps.append(
                PipelineStep(
                    "annotate_allcategories_on_rawframe",
                    lambda: tracker.annotate_allcategories_on_rawframe(
                        raw_frame_dir=cfg.input.raw_frame_dir,
                        output_dir=annotations.all_category_output_dir,
                        mask_alpha=annotations.mask_alpha,
                        show_centroid=annotations.all_category_show_centroid,
                        label_ids=annotations.all_category_label_ids,
                        max_dist=tracking_max_dist,
                        frame_step=annotations.frame_step,
                    ),
                )
            )

        if annotations.save_boundary_pair_id_raw_frames:
            steps.append(
                PipelineStep(
                    "annotate_boundary_pair_ids_on_rawframe",
                    lambda: tracker.annotate_boundary_pair_ids_on_rawframe(
                        raw_frame_dir=cfg.input.raw_frame_dir,
                        output_dir=annotations.boundary_pair_output_dir,
                        mask_alpha=annotations.mask_alpha,
                        show_centroid=annotations.all_category_show_centroid,
                        max_dist=tracking_max_dist,
                        frame_step=annotations.frame_step,
                    ),
                )
            )

        if plots.save_evolution:
            steps.append(
                PipelineStep(
                    "plot_evolution",
                    lambda: tracker.plot_evolution(step=plots.evolution_step),
                )
            )
        if plots.save_centroid_trajectories:
            steps.append(
                PipelineStep(
                    "plot_centroid_trajectories",
                    lambda: tracker.plot_centroid_trajectories(
                        max_dist=tracking_max_dist,
                        annotate_ids_max=plots.annotate_ids_max,
                    ),
                )
            )
        if plots.save_area_trajectories:
            steps.append(
                PipelineStep(
                    "plot_area_trajectories",
                    lambda: tracker.plot_area_trajectories(
                        max_dist=tracking_max_dist,
                        min_track_length=plots.min_track_length,
                        debug_stats=plots.debug_stats,
                        max_plot_tracks=plots.max_tracks,
                        max_legend_items=plots.max_legend_items,
                        annotate_ids_max=plots.annotate_ids_max,
                    ),
                )
            )
        if plots.save_frame_count_area:
            steps.append(
                PipelineStep(
                    "plot_frame_instance_count_and_total_area",
                    tracker.plot_frame_instance_count_and_total_area,
                )
            )
        if plots.save_area_delta:
            steps.append(
                PipelineStep(
                    "plot_area_delta_vs_frame",
                    lambda: tracker.plot_area_delta_vs_frame(
                        per_frame=plots.area_delta_per_frame,
                        reducer=plots.area_delta_reducer,
                    ),
                )
            )
        if plots.save_velocity_trajectories:
            steps.append(
                PipelineStep(
                    "plot_velocity_trajectories",
                    lambda: tracker.plot_velocity_trajectories(
                        max_dist=tracking_max_dist,
                        min_track_length=plots.min_track_length,
                        frame_interval_s=plots.frame_interval_s,
                        bin_size_frames=plots.velocity_bin_size_frames,
                        debug_stats=plots.debug_stats,
                        max_plot_tracks=plots.max_tracks,
                        max_legend_items=plots.max_legend_items,
                        annotate_ids_max=plots.annotate_ids_max,
                    ),
                )
            )
        if plots.save_boundary_distance_plots:
            steps.append(
                PipelineStep(
                    "plot_boundary_distances_vs_frame",
                    lambda: tracker.plot_boundary_distances_vs_frame(
                        max_dist=tracking_max_dist,
                        min_pair_length=plots.min_track_length,
                        max_plot_pairs=plots.max_tracks,
                        max_legend_items=plots.max_legend_items,
                        debug_stats=plots.debug_stats,
                    ),
                )
            )
        if cfg.output.export_csv_results:
            steps.append(
                PipelineStep(
                    "validate_exported_csv_ids",
                    tracker.validate_exported_csv_ids,
                )
            )
        return steps

    def _timed_step(self, step: PipelineStep) -> Any:
        started = time.perf_counter()
        self._log(f"[time] start {step.name}")
        result = step.action()
        duration = time.perf_counter() - started
        self._log(f"[time] done  {step.name}: {duration:.2f}s")
        return result
