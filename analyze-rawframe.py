"""Backward-compatible command line entry point for raw-frame analysis."""

from rawframe_analysis.cli import main

__all__ = ["GasTracker", "main"]


def __getattr__(name):
    if name == "GasTracker":
        from rawframe_analysis.tracker import GasTracker

        return GasTracker
    raise AttributeError(name)


if __name__ == "__main__":
    raise SystemExit(main())
