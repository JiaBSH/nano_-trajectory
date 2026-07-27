"""Typed configuration and JSON persistence for raw-frame analysis."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, TypeVar


class ConfigError(ValueError):
    """Raised when a configuration file is invalid or incomplete."""


@dataclass(frozen=True, slots=True)
class InputConfig:
    json_dir: str = ""
    scale_csv: str | None = None
    image_path: str | None = None
    raw_frame_dir: str | None = None
    scale_value_nm: float = 20.0
    nm_per_px: float | None = None
    strict_scale_match: bool = False


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    target_category: str = "gas"
    pin_category: str = "pin"
    pin_reference_enabled: bool = False
    skip_frames_without_pin: bool = True
    max_particle_pin_distance_nm: float | None = None
    fastplot_enabled: bool = True
    compute_diameter_height_enabled: bool = False


@dataclass(frozen=True, slots=True)
class OutputConfig:
    root: str | None = None
    export_csv_results: bool = True
    export_max_dist_nm: float = 50.0
    export_use_display_id: bool = True


@dataclass(frozen=True, slots=True)
class AnnotationConfig:
    save_target_raw_frames: bool = False
    save_all_category_raw_frames: bool = False
    frame_step: int = 1
    mask_alpha: int = 120
    target_output_dir: str | None = None
    all_category_output_dir: str = "annotated_allcat_rawframe"
    target_label_ids: bool = True
    all_category_label_ids: bool = False
    all_category_show_centroid: bool = False


@dataclass(frozen=True, slots=True)
class PlotConfig:
    save_evolution: bool = True
    save_centroid_trajectories: bool = True
    save_area_trajectories: bool = True
    save_frame_count_area: bool = True
    save_area_delta: bool = True
    save_velocity_trajectories: bool = True
    max_dist_nm: float = 20.0
    min_track_length: int = 0
    max_tracks: int = 500
    max_legend_items: int = 60
    annotate_ids_max: int = 80
    debug_stats: bool = True
    evolution_step: int = 2
    area_delta_per_frame: bool = True
    area_delta_reducer: str = "sum"
    frame_interval_s: float = 1.0 / 30.0
    velocity_bin_size_frames: int = 1


_SectionT = TypeVar("_SectionT")
_CURRENT_CONFIG_VERSION = 2


def _rename_legacy_option(
    section_name: str,
    section: dict[str, Any],
    old_name: str,
    new_name: str,
) -> None:
    if old_name not in section:
        return
    if new_name in section:
        raise ConfigError(
            f"'{section_name}' cannot contain both '{old_name}' and '{new_name}'"
        )
    section[new_name] = section.pop(old_name)


def _matches_derived_name(value: object, expected: str) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized == expected


def _migrate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade legacy version-1 gas-specific option names to version 2."""
    migrated = deepcopy(dict(raw))
    version = migrated.get("config_version", 1)
    if version == _CURRENT_CONFIG_VERSION:
        return migrated
    if version != 1:
        raise ConfigError(
            f"Unsupported config_version: {version!r}; expected 1 or {_CURRENT_CONFIG_VERSION}"
        )

    analysis = migrated.setdefault("analysis", {})
    annotations = migrated.setdefault("annotations", {})
    output = migrated.setdefault("output", {})
    if isinstance(analysis, dict):
        _rename_legacy_option("analysis", analysis, "gas_category", "target_category")
    if isinstance(annotations, dict):
        _rename_legacy_option(
            "annotations",
            annotations,
            "save_gas_raw_frames",
            "save_target_raw_frames",
        )
        _rename_legacy_option(
            "annotations", annotations, "gas_output_dir", "target_output_dir"
        )
        _rename_legacy_option(
            "annotations", annotations, "gas_label_ids", "target_label_ids"
        )

    if isinstance(analysis, dict):
        category = str(analysis.get("target_category", "gas"))
        pin_suffix = (
            "_pin_relative" if analysis.get("pin_reference_enabled", False) else ""
        )
        if isinstance(output, dict) and _matches_derived_name(
            output.get("root"), f"{category}{pin_suffix}"
        ):
            output["root"] = None
        if isinstance(annotations, dict) and _matches_derived_name(
            annotations.get("target_output_dir"),
            f"annotated_{category}_rawframe",
        ):
            annotations["target_output_dir"] = None

    migrated["config_version"] = _CURRENT_CONFIG_VERSION
    return migrated


def _decode_section(
    section_name: str,
    section_type: type[_SectionT],
    raw: object,
) -> _SectionT:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'{section_name}' must be a JSON object")

    known = {field.name for field in fields(section_type)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(
            f"Unknown option(s) in '{section_name}': {', '.join(unknown)}"
        )

    try:
        return section_type(**dict(raw))
    except TypeError as exc:
        raise ConfigError(f"Invalid '{section_name}' section: {exc}") from exc


def _resolve_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    raw_path = Path(value).expanduser()
    if not raw_path.is_absolute():
        raw_path = base_dir / raw_path
    return os.fspath(raw_path.resolve())


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"'{name}' must be true or false")


def _require_number(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> None:
    if not _is_number(value):
        raise ConfigError(f"'{name}' must be a number")
    numeric_value = float(value)
    if strictly_positive and numeric_value <= 0:
        raise ConfigError(f"'{name}' must be greater than 0")
    if minimum is not None and numeric_value < minimum:
        raise ConfigError(f"'{name}' must be at least {minimum}")


@dataclass(frozen=True, slots=True)
class RawFrameConfig:
    """Complete, validated configuration used by one analysis run."""

    config_version: int = _CURRENT_CONFIG_VERSION
    input: InputConfig = InputConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    output: OutputConfig = OutputConfig()
    annotations: AnnotationConfig = AnnotationConfig()
    plots: PlotConfig = PlotConfig()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RawFrameConfig":
        if not isinstance(raw, Mapping):
            raise ConfigError("The configuration root must be a JSON object")
        raw = _migrate_config(raw)

        known = {
            "config_version",
            "input",
            "analysis",
            "output",
            "annotations",
            "plots",
        }
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ConfigError(f"Unknown top-level section(s): {', '.join(unknown)}")

        version = raw.get("config_version", _CURRENT_CONFIG_VERSION)
        if version != _CURRENT_CONFIG_VERSION:
            raise ConfigError(
                f"Unsupported config_version: {version!r}; expected {_CURRENT_CONFIG_VERSION}"
            )

        config = cls(
            config_version=version,
            input=_decode_section("input", InputConfig, raw.get("input", {})),
            analysis=_decode_section(
                "analysis", AnalysisConfig, raw.get("analysis", {})
            ),
            output=_decode_section("output", OutputConfig, raw.get("output", {})),
            annotations=_decode_section(
                "annotations", AnnotationConfig, raw.get("annotations", {})
            ),
            plots=_decode_section("plots", PlotConfig, raw.get("plots", {})),
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_relative_to(
        self, config_path: str | os.PathLike[str]
    ) -> "RawFrameConfig":
        """Resolve every filesystem path relative to the source config file."""
        base_dir = Path(config_path).resolve().parent
        category = self.analysis.target_category
        output_root = self.output.root
        if not output_root:
            suffix = "_pin_relative" if self.analysis.pin_reference_enabled else ""
            output_root = f"{category}{suffix}"
        else:
            output_root = output_root.replace("{category}", category)

        target_output_dir = self.annotations.target_output_dir
        if not target_output_dir:
            target_output_dir = f"annotated_{category}_rawframe"
        else:
            target_output_dir = target_output_dir.replace("{category}", category)

        resolved = replace(
            self,
            input=replace(
                self.input,
                json_dir=_resolve_path(self.input.json_dir, base_dir) or "",
                scale_csv=_resolve_path(self.input.scale_csv, base_dir),
                image_path=_resolve_path(self.input.image_path, base_dir),
                raw_frame_dir=_resolve_path(self.input.raw_frame_dir, base_dir),
            ),
            output=replace(
                self.output,
                root=_resolve_path(output_root, base_dir),
            ),
            annotations=replace(
                self.annotations,
                target_output_dir=target_output_dir,
            ),
        )
        resolved.validate()
        return resolved

    def validate(self) -> None:
        if not isinstance(self.input.json_dir, str) or not self.input.json_dir.strip():
            raise ConfigError("'input.json_dir' must be a non-empty path")
        for name in ("scale_csv", "image_path", "raw_frame_dir"):
            value = getattr(self.input, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"'input.{name}' must be null or a non-empty path")

        _require_number(
            "input.scale_value_nm", self.input.scale_value_nm, strictly_positive=True
        )
        if self.input.nm_per_px is not None:
            _require_number(
                "input.nm_per_px", self.input.nm_per_px, strictly_positive=True
            )
        _require_bool("input.strict_scale_match", self.input.strict_scale_match)

        if (
            not isinstance(self.analysis.target_category, str)
            or not self.analysis.target_category.strip()
        ):
            raise ConfigError("'analysis.target_category' must be a non-empty string")
        if (
            not isinstance(self.analysis.pin_category, str)
            or not self.analysis.pin_category.strip()
        ):
            raise ConfigError("'analysis.pin_category' must be a non-empty string")
        for name in (
            "pin_reference_enabled",
            "skip_frames_without_pin",
            "fastplot_enabled",
            "compute_diameter_height_enabled",
        ):
            _require_bool(f"analysis.{name}", getattr(self.analysis, name))
        if self.analysis.max_particle_pin_distance_nm is not None:
            _require_number(
                "analysis.max_particle_pin_distance_nm",
                self.analysis.max_particle_pin_distance_nm,
                strictly_positive=True,
            )

        if self.output.root is not None and (
            not isinstance(self.output.root, str) or not self.output.root.strip()
        ):
            raise ConfigError("'output.root' must be null or a non-empty path")
        _require_bool("output.export_csv_results", self.output.export_csv_results)
        _require_bool("output.export_use_display_id", self.output.export_use_display_id)
        _require_number(
            "output.export_max_dist_nm",
            self.output.export_max_dist_nm,
            strictly_positive=True,
        )

        for name in ("save_target_raw_frames", "save_all_category_raw_frames"):
            _require_bool(f"annotations.{name}", getattr(self.annotations, name))
        _require_number(
            "annotations.frame_step", self.annotations.frame_step, minimum=1
        )
        if not isinstance(self.annotations.frame_step, int):
            raise ConfigError("'annotations.frame_step' must be an integer")
        _require_number(
            "annotations.mask_alpha", self.annotations.mask_alpha, minimum=0
        )
        if (
            not isinstance(self.annotations.mask_alpha, int)
            or self.annotations.mask_alpha > 255
        ):
            raise ConfigError(
                "'annotations.mask_alpha' must be an integer from 0 to 255"
            )
        for name in (
            "target_label_ids",
            "all_category_label_ids",
            "all_category_show_centroid",
        ):
            _require_bool(f"annotations.{name}", getattr(self.annotations, name))
        for name in ("target_output_dir", "all_category_output_dir"):
            value = getattr(self.annotations, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(
                    f"'annotations.{name}' must be null or a non-empty string"
                )

        for name in (
            "save_evolution",
            "save_centroid_trajectories",
            "save_area_trajectories",
            "save_frame_count_area",
            "save_area_delta",
            "save_velocity_trajectories",
            "debug_stats",
            "area_delta_per_frame",
        ):
            _require_bool(f"plots.{name}", getattr(self.plots, name))
        _require_number(
            "plots.max_dist_nm", self.plots.max_dist_nm, strictly_positive=True
        )
        _require_number(
            "plots.frame_interval_s",
            self.plots.frame_interval_s,
            strictly_positive=True,
        )
        for name, minimum in (
            ("min_track_length", 0),
            ("max_tracks", 0),
            ("max_legend_items", 0),
            ("annotate_ids_max", 0),
            ("evolution_step", 1),
            ("velocity_bin_size_frames", 1),
        ):
            value = getattr(self.plots, name)
            _require_number(f"plots.{name}", value, minimum=minimum)
            if not isinstance(value, int):
                raise ConfigError(f"'plots.{name}' must be an integer")
        if self.plots.area_delta_reducer not in {"sum", "mean"}:
            raise ConfigError("'plots.area_delta_reducer' must be 'sum' or 'mean'")

    def validate_input_paths(self) -> None:
        """Check paths that are required for the selected run options."""
        json_dir = Path(self.input.json_dir)
        if not json_dir.is_dir():
            raise ConfigError(f"JSON annotation directory does not exist: {json_dir}")

        if (
            self.input.scale_csv is not None
            and not Path(self.input.scale_csv).is_file()
        ):
            raise ConfigError(f"Scale CSV does not exist: {self.input.scale_csv}")
        if (
            self.input.image_path is not None
            and not Path(self.input.image_path).exists()
        ):
            raise ConfigError(f"Image path does not exist: {self.input.image_path}")

        annotations_enabled = (
            self.annotations.save_target_raw_frames
            or self.annotations.save_all_category_raw_frames
        )
        if annotations_enabled:
            if self.input.raw_frame_dir is None:
                raise ConfigError(
                    "'input.raw_frame_dir' is required when raw-frame annotations are enabled"
                )
            if not Path(self.input.raw_frame_dir).is_dir():
                raise ConfigError(
                    f"Raw-frame image directory does not exist: {self.input.raw_frame_dir}"
                )


class JsonConfigRepository:
    """Read and write the JSON representation of :class:`RawFrameConfig`."""

    def load(self, path: str | os.PathLike[str]) -> RawFrameConfig:
        source = Path(path)
        try:
            with source.open("r", encoding="utf-8-sig") as handle:
                raw = json.load(handle)
        except FileNotFoundError as exc:
            raise ConfigError(f"Configuration file does not exist: {source}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"Invalid JSON in {source} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc

        return RawFrameConfig.from_dict(raw)

    def save(self, config: RawFrameConfig, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                temp_name = handle.name
            os.replace(temp_name, target)
        finally:
            if temp_name is not None and os.path.exists(temp_name):
                os.unlink(temp_name)
        return target


@dataclass(frozen=True, slots=True)
class ConfigSnapshot:
    latest: Path
    archived: Path


class ConfigSnapshotWriter:
    """Persist both the latest and an immutable per-run config snapshot."""

    def __init__(self, repository: JsonConfigRepository | None = None) -> None:
        self._repository = repository or JsonConfigRepository()

    def write(self, config: RawFrameConfig) -> ConfigSnapshot:
        if not config.output.root:
            raise ConfigError(
                "Output root must be resolved before saving a config snapshot"
            )

        output_root = Path(config.output.root)
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        latest = output_root / "run_config.json"
        archived = output_root / "run_configs" / f"run_config_{timestamp}.json"
        self._repository.save(config, archived)
        self._repository.save(config, latest)
        return ConfigSnapshot(latest=latest, archived=archived)
