"""Public tracker assembled from focused analysis capabilities."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from .annotations import AnnotationMixin
from .boundary_distance_plots import BoundaryDistancePlotMixin
from .exporting import CsvExportMixin
from .geometry import GeometryMixin
from .inputs import InputMixin
from .plot_style import PlotStyleMixin
from .postprocessing import InstancePostprocessingMixin
from .processing import FrameProcessingMixin
from .summary_plots import SummaryPlotMixin
from .tracking import ObjectTrackingMixin
from .trajectory_plots import TrajectoryPlotMixin


class GasTracker(
    InputMixin,
    GeometryMixin,
    InstancePostprocessingMixin,
    FrameProcessingMixin,
    ObjectTrackingMixin,
    CsvExportMixin,
    AnnotationMixin,
    PlotStyleMixin,
    BoundaryDistancePlotMixin,
    TrajectoryPlotMixin,
    SummaryPlotMixin,
):
    """Analyze segmented gas objects while preserving the original public API."""

    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        json_dir,
        image_path=None,
        scale_csv=None,
        scale_value_nm=20.0,
        nm_per_px=None,
        strict_scale_match=False,
        target_category="gas",
        pin_category="pin",
        pin_reference_enabled=False,
        skip_frames_without_pin=True,
        pin_centroid_smoothing_alpha=0.25,
        max_particle_pin_distance_nm=None,
        fastplot_enabled=True,
        compute_diameter_height_enabled=True,
        output_root=None,
        gas_category=None,
        compute_boundary_distances_enabled=False,
        particle_category="nanocluster",
        droplet_category="nanodroplet",
        instance_overlap_postprocess_enabled=True,
        same_category_containment_threshold=0.85,
    ):
        target_category = self._resolve_legacy_category(
            target_category=target_category,
            gas_category=gas_category,
        )
        self._set_options(
            json_dir=json_dir,
            image_path=image_path,
            scale_csv=scale_csv,
            scale_value_nm=scale_value_nm,
            nm_per_px=nm_per_px,
            strict_scale_match=strict_scale_match,
            target_category=target_category,
            pin_category=pin_category,
            pin_reference_enabled=pin_reference_enabled,
            skip_frames_without_pin=skip_frames_without_pin,
            pin_centroid_smoothing_alpha=pin_centroid_smoothing_alpha,
            max_particle_pin_distance_nm=max_particle_pin_distance_nm,
            fastplot_enabled=fastplot_enabled,
            compute_diameter_height_enabled=compute_diameter_height_enabled,
            compute_boundary_distances_enabled=compute_boundary_distances_enabled,
            particle_category=particle_category,
            droplet_category=droplet_category,
            instance_overlap_postprocess_enabled=instance_overlap_postprocess_enabled,
            same_category_containment_threshold=same_category_containment_threshold,
        )
        self._initialize_run_state()
        self._prepare_output_directory(output_root)
        self._initialize_inputs()
        self._initialize_scale_state()
        self._initialize_record_buffers()
        self._initialize_spatial_reference()
        self._initialize_image_dimensions()
        self._configure_matplotlib_fonts()

    def _set_options(
        self,
        *,
        json_dir,
        image_path,
        scale_csv,
        scale_value_nm,
        nm_per_px,
        strict_scale_match,
        target_category,
        pin_category,
        pin_reference_enabled,
        skip_frames_without_pin,
        pin_centroid_smoothing_alpha,
        max_particle_pin_distance_nm,
        fastplot_enabled,
        compute_diameter_height_enabled,
        compute_boundary_distances_enabled,
        particle_category,
        droplet_category,
        instance_overlap_postprocess_enabled,
        same_category_containment_threshold,
    ):
        """Normalize constructor arguments into stable tracker options."""
        self.json_dir = json_dir
        self.image_path = os.fspath(image_path) if image_path else None
        self.image_dir = None
        self.scale_csv = scale_csv
        self.scale_value_nm = float(scale_value_nm)
        self.fixed_nm_per_px = float(nm_per_px) if nm_per_px is not None else None
        self.strict_scale_match = bool(strict_scale_match)
        self.target_category = target_category
        self.pin_category = pin_category
        self.pin_reference_enabled = bool(pin_reference_enabled)
        self.skip_frames_without_pin = bool(skip_frames_without_pin)
        self.pin_centroid_smoothing_alpha = self._unit_interval_float(
            "pin_centroid_smoothing_alpha", pin_centroid_smoothing_alpha
        )
        self.max_particle_pin_distance_nm = self._positive_optional_float(
            "max_particle_pin_distance_nm", max_particle_pin_distance_nm
        )
        self.fastplot_enabled = bool(fastplot_enabled)
        self.compute_diameter_height_enabled = bool(compute_diameter_height_enabled)
        self.compute_boundary_distances_enabled = bool(
            compute_boundary_distances_enabled
        )
        self.particle_category = str(particle_category).strip()
        self.droplet_category = str(droplet_category).strip()
        if not self.particle_category or not self.droplet_category:
            raise ValueError("particle_category and droplet_category must be non-empty")
        if self.particle_category == self.droplet_category:
            raise ValueError("particle_category and droplet_category must be different")
        self.instance_overlap_postprocess_enabled = bool(
            instance_overlap_postprocess_enabled
        )
        self.same_category_containment_threshold = self._unit_interval_float(
            "same_category_containment_threshold",
            same_category_containment_threshold,
        )

    @staticmethod
    def _resolve_legacy_category(*, target_category, gas_category):
        """Accept the old keyword while keeping target_category canonical."""
        if gas_category is None:
            return target_category
        if target_category != "gas" and target_category != gas_category:
            raise ValueError(
                "target_category and legacy gas_category specify different values"
            )
        return gas_category

    @property
    def gas_category(self):
        """Backward-compatible alias for code written before config version 2."""
        return self.target_category

    @gas_category.setter
    def gas_category(self, value):
        self.target_category = value

    @staticmethod
    def _positive_optional_float(name, value):
        if value is None:
            return None
        normalized = float(value)
        if normalized <= 0:
            raise ValueError(f"{name} must be positive, got {normalized}")
        return normalized

    @staticmethod
    def _unit_interval_float(name, value):
        normalized = float(value)
        if not 0.0 < normalized <= 1.0:
            raise ValueError(f"{name} must be in (0, 1], got {normalized}")
        return normalized

    def _initialize_run_state(self):
        """Reset counters and caches that belong to one analysis run."""
        self.pin_reference_records = []
        self.filtered_far_particle_records = []
        self.filtered_far_particle_count = 0
        self.skipped_no_pin_frames = 0
        self.processed_frame_count = 0
        self._object_detections_by_frame_cache = None
        self._object_detections_by_frame_cache_records = None
        self._object_detections_by_frame_cache_record_count = 0
        self._event_id_series_cache = {}
        self._postprocessed_frame_cache = {}
        self.instance_postprocess_records = []
        self.same_category_suppressed_count = 0
        self.cross_category_suppressed_count = 0

    def _prepare_output_directory(self, output_root):
        if output_root is None:
            output_root = (
                f"{self.target_category}_pin_relative"
                if self.pin_reference_enabled
                else self.target_category
            )
        self.output_root = os.fspath(output_root)
        os.makedirs(self.output_root, exist_ok=True)

    def _initialize_inputs(self):
        self.json_files = self._load_and_sort_jsons()
        self.image_path, self.image_dir = self._resolve_image_input(self.image_path)

    def _initialize_scale_state(self):
        """Load fixed or per-frame physical scale information."""
        self.scale_map = {}
        self.fallback_nm_per_px = None
        self.max_nm_per_px = None
        self.min_nm_per_px = None
        self._warned_no_scale_csv = False
        self._warned_missing_scale_match = False

        if self.fixed_nm_per_px is not None:
            if self.fixed_nm_per_px <= 0:
                raise ValueError(
                    f"nm_per_px must be positive, got {self.fixed_nm_per_px}"
                )
            self.fallback_nm_per_px = self.fixed_nm_per_px
            self.max_nm_per_px = self.fixed_nm_per_px
            self.min_nm_per_px = self.fixed_nm_per_px
            return
        if self.scale_csv is None:
            return

        annotated_stems = {Path(name).stem for name in self.json_files}
        self.scale_map = self._load_nm_per_px_map(
            self.scale_csv,
            default_scale_value_nm=self.scale_value_nm,
            allowed_stems=annotated_stems,
        )
        if not self.scale_map:
            raise ValueError(
                "Scale CSV provided but no usable rows matched annotated JSON frames: "
                f"{self.scale_csv}"
            )

        values = np.asarray(list(self.scale_map.values()), dtype=np.float64)
        self.fallback_nm_per_px = float(np.median(values))
        self.max_nm_per_px = float(np.max(values))
        self.min_nm_per_px = float(np.min(values))

    def _initialize_record_buffers(self):
        """Create result buffers; coordinates use nm and areas use nm²."""
        self.area_records = []
        self.contour_records = []
        self.centroid_records = []
        self.object_records = []
        self.diameter_height_records = []
        self.particle_particle_distance_records = []
        self.particle_droplet_distance_records = []
        self.particle_nearest_distance_records = []
        self.boundary_particle_records = []
        self.boundary_droplet_records = []

    def _initialize_spatial_reference(self):
        self.ref_pin_centroid = None
        self.stabilized_pin_centroid = None
        self.last_shift = np.zeros(2)

    def _initialize_image_dimensions(self):
        self.W, self.H = None, None
        if self.image_path:
            with Image.open(self.image_path) as image:
                self.W, self.H = image.size
