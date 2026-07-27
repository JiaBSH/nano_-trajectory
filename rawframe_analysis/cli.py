"""Command line interface for raw-frame analysis."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from .config import ConfigError, JsonConfigRepository
from .pipeline import AnalysisPipeline, TrackerFactory


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "analyze-rawframe.config.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze segmented raw frames using a versioned JSON configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.fspath(DEFAULT_CONFIG_PATH),
        help=f"JSON config path (default: {DEFAULT_CONFIG_PATH.name})",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and input paths without running analysis.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    tracker_factory: TrackerFactory | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repository = JsonConfigRepository()

    try:
        config = repository.load(args.config).resolved_relative_to(args.config)
        config.validate_input_paths()
    except ConfigError as exc:
        parser.exit(2, f"Configuration error: {exc}\n")

    if args.validate_only:
        print(f"Configuration is valid: {Path(args.config).resolve()}")
        print(f"Output root: {config.output.root}")
        return 0

    if tracker_factory is None:
        from .tracker import GasTracker

        tracker_factory = GasTracker

    AnalysisPipeline(config, tracker_factory).run()
    return 0
