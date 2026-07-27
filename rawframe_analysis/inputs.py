"""Input discovery and physical-scale loading for raw-frame analysis."""

from __future__ import annotations

import csv
import os
from pathlib import Path


class InputMixin:
    """Provide input discovery and physical-scale loading for raw-frame analysis."""

    def _resolve_image_input(self, image_path):
        """Accept either a single reference image or a directory of frame images."""
        if not image_path:
            return None, None

        if os.path.isdir(image_path):
            image_dir = image_path
            ref_image = self._find_reference_image(image_dir)
            if ref_image is None:
                print(
                    f"[warn] image_path points to a directory, but no image was found: {image_dir}. "
                    "Plots will fall back to data bounds."
                )
            return ref_image, image_dir

        return image_path, os.path.dirname(image_path)

    def _find_reference_image(self, image_dir):
        """Find the first image matching the sorted JSON frames, then fall back to any image."""
        for json_name in self.json_files:
            frame_name = Path(json_name).stem
            for ext in self.IMAGE_EXTS:
                candidate = os.path.join(image_dir, frame_name + ext)
                if os.path.isfile(candidate):
                    return candidate

        try:
            for name in sorted(os.listdir(image_dir)):
                if Path(name).suffix.lower() in self.IMAGE_EXTS:
                    candidate = os.path.join(image_dir, name)
                    if os.path.isfile(candidate):
                        return candidate
        except OSError:
            return None

        return None

    @staticmethod
    def _parse_scale_value_to_nm(scale_value, unit):
        if scale_value is None:
            return None
        if unit is None:
            return float(scale_value)
        u = str(unit).strip().lower()
        v = float(scale_value)
        if u in {"nm", "nanometer", "nanometers"}:
            return v
        if u in {"um", "µm", "micrometer", "micrometers"}:
            return v * 1000.0
        if u in {"mm"}:
            return v * 1_000_000.0
        return v

    @classmethod
    def _load_nm_per_px_map(
        cls, csv_path, default_scale_value_nm=20.0, allowed_stems=None
    ):
        """Load per-image nm/px from a scalebar CSV.

        Supports:
        - minimal CSV: image,pixel_length
        - yolo_easyocr output: image,scale_value,unit,pixel_length,ratio,...

        Keying:
        - uses image basename stem, e.g. '..._000000000003'
        - when allowed_stems is provided, rows for unannotated frames are ignored
        """
        csv_path = str(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Scale CSV not found: {csv_path}. "
                "Please provide a CSV with columns 'image' and 'pixel_length'."
            )

        nm_per_px = {}
        allowed_stems = set(allowed_stems) if allowed_stems is not None else None
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = (row.get("image") or row.get("img") or "").strip()
                px_len = row.get("pixel_length")
                if img == "" or px_len in (None, ""):
                    continue

                try:
                    pixel_length = float(px_len)
                except Exception:
                    continue
                if pixel_length <= 0:
                    continue

                scale_value = row.get("scale_value")
                unit = row.get("unit")
                scale_nm = None
                if scale_value not in (None, ""):
                    try:
                        scale_nm = cls._parse_scale_value_to_nm(scale_value, unit)
                    except Exception:
                        scale_nm = None
                if scale_nm is None:
                    scale_nm = float(default_scale_value_nm)

                stem = Path(img).stem
                if allowed_stems is not None and stem not in allowed_stems:
                    continue
                nm_per_px[stem] = float(scale_nm) / float(pixel_length)

        return nm_per_px

    def _nm_per_px_for_frame(self, frame_name):
        if self.fixed_nm_per_px is not None:
            return float(self.fixed_nm_per_px)
        if self.scale_csv is None:
            # No scale CSV: keep pipeline running in pixel-space (1 px = 1 pseudo-nm unit).
            if not self._warned_no_scale_csv:
                print(
                    "[warn] scale_csv is not set. Continue with fallback nm_per_px=1.0 "
                    "(numerical values are pixel-scale, not real nm)."
                )
                self._warned_no_scale_csv = True
            return 1.0
        v = self.scale_map.get(frame_name)
        if v is not None:
            return float(v)
        if self.strict_scale_match:
            raise KeyError(
                f"No scale entry for frame '{frame_name}' in {self.scale_csv}"
            )
        # Non-strict mode: do not skip frame; use dataset-level fallback if available.
        if self.fallback_nm_per_px is not None:
            if not self._warned_missing_scale_match:
                print(
                    "[warn] Some frames have no matching scale in CSV. "
                    f"Using fallback median nm_per_px={self.fallback_nm_per_px:.6f}."
                )
                self._warned_missing_scale_match = True
            return float(self.fallback_nm_per_px)
        if not self._warned_missing_scale_match:
            print(
                "[warn] No matching scale and no fallback available. "
                "Use nm_per_px=1.0 (pixel-scale units)."
            )
            self._warned_missing_scale_match = True
        return 1.0

    def _load_and_sort_jsons(self):
        import re

        files = [f for f in os.listdir(self.json_dir) if f.endswith(".json")]

        def _sort_key(name):
            stem = Path(name).stem
            m = re.search(r"(\d+)$", stem)
            if m is not None:
                # Keep original behavior for names ending with numeric frame id.
                return (0, int(m.group(1)), stem.lower())
            # Fallback: non-numeric names are sorted lexicographically after numeric ones.
            return (1, 0, stem.lower())

        files.sort(key=_sort_key)
        return files
