"""Raw-frame segmentation analysis package."""

from .config import ConfigError, RawFrameConfig
from .pipeline import AnalysisPipeline, RunResult

__all__ = [
    "AnalysisPipeline",
    "ConfigError",
    "GasTracker",
    "RawFrameConfig",
    "RunResult",
]


def __getattr__(name):
    """Keep the public GasTracker name without importing plotting dependencies eagerly."""
    if name == "GasTracker":
        from .tracker import GasTracker

        return GasTracker
    raise AttributeError(name)
