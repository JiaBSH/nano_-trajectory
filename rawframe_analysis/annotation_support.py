"""Shared annotation argument validation."""

from __future__ import annotations

import os


class AnnotationSupportMixin:
    """Provide shared annotation argument validation."""

    @staticmethod
    def _normalize_frame_step(frame_step):
        if frame_step is None:
            return 1
        step = int(frame_step)
        if step < 1:
            raise ValueError("frame_step must be >= 1")
        return step

    def _resolve_annotation_output_dir(self, output_dir, default_name):
        """Return an existing output directory rooted under this run by default."""
        resolved = output_dir or default_name
        if not os.path.isabs(resolved):
            resolved = os.path.join(self.output_root, resolved)
        os.makedirs(resolved, exist_ok=True)
        return resolved
