"""Compatibility entry point for the merged raw-frame analyzer."""

import importlib.util
import runpy
from pathlib import Path


BASE_SCRIPT = Path(__file__).resolve().with_name("analyze-rawframe.py")


def _load_base_module():
    spec = importlib.util.spec_from_file_location("analyze_rawframe_merged", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load merged analyzer: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    runpy.run_path(str(BASE_SCRIPT), run_name="__main__")
else:
    _base = _load_base_module()
    GasTracker = _base.GasTracker
