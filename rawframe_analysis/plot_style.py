"""Shared Matplotlib style and axis helpers."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


class PlotStyleMixin:
    """Provide shared matplotlib style and axis helpers."""

    @staticmethod
    def _configure_matplotlib_fonts():
        """Configure Matplotlib fonts for Chinese text.

        If suitable CJK fonts aren't available, Matplotlib will fall back and may show tofu boxes.
        """
        preferred = [
            "Microsoft YaHei",  # 微软雅黑
            "SimHei",  # 黑体
            "PingFang SC",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]

        try:
            available = {f.name for f in fm.fontManager.ttflist}
            chosen = [name for name in preferred if name in available]
            if chosen:
                plt.rcParams["font.sans-serif"] = chosen
        except Exception:
            # best-effort: still set a reasonable default list
            plt.rcParams["font.sans-serif"] = preferred

        # Global plotting font sizes: keep all generated chart text consistently larger.
        plt.rcParams["font.size"] = 17
        plt.rcParams["axes.titlesize"] = 20
        plt.rcParams["axes.labelsize"] = 17
        plt.rcParams["xtick.labelsize"] = 15
        plt.rcParams["ytick.labelsize"] = 15
        plt.rcParams["legend.fontsize"] = 14
        plt.rcParams["figure.titlesize"] = 20

        plt.rcParams["axes.unicode_minus"] = False

    def _set_nm_axes(self, ax, point_arrays=None):
        """Set plot axes from image size when available, otherwise from plotted nm data."""
        scale = float(self.max_nm_per_px) if self.max_nm_per_px is not None else 1.0
        force_data_bounds = bool(getattr(self, "pin_reference_enabled", False))

        if self.W is not None and self.H is not None and not force_data_bounds:
            ax.set_xlim(0, self.W * scale * 1.5)
            ax.set_ylim(self.H * scale, 0)
        else:
            arrays = []
            for pts in point_arrays or []:
                arr = np.asarray(pts, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
                    continue
                arr = arr[:, :2]
                arr = arr[np.isfinite(arr).all(axis=1)]
                if arr.shape[0] > 0:
                    arrays.append(arr)

            if not arrays:
                return False

            pts = np.vstack(arrays)
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            span = max(float(max_x - min_x), float(max_y - min_y))
            pad = max(span * 0.05, 1.0)
            ax.set_xlim(float(min_x - pad), float(max_x + pad))
            ax.set_ylim(float(max_y + pad), float(min_y - pad))

        if force_data_bounds:
            ax.set_xlabel("x relative to pin centroid (nm)")
            ax.set_ylabel("y relative to pin centroid (nm)")
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.25)
            ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.25)
        else:
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")
        ax.set_aspect("equal", adjustable="box")
        return True
