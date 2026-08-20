"""Frame-local cleanup for overlapping instance-segmentation masks."""

from __future__ import annotations

from copy import copy
from math import ceil, floor

import numpy as np
from PIL import Image, ImageDraw


class InstancePostprocessingMixin:
    """Suppress duplicate and conflicting particle/droplet masks.

    Model predictions can contain a smaller mask almost completely inside a
    second mask for the same physical object.  Tracking such raw predictions
    creates false births and many short-lived IDs.  This mixin performs
    frame-local mask NMS before any measurement, tracking, export, or drawing.
    """

    def _postprocess_frame_instances(self, data, frame_name=None):
        if not bool(self.instance_overlap_postprocess_enabled):
            return data

        cache_key = None if frame_name is None else str(frame_name)
        if cache_key is not None and cache_key in self._postprocessed_frame_cache:
            return self._postprocessed_frame_cache[cache_key]

        raw_objects = list(data.get("objects", []))
        objects = []
        valid_points = {}
        selected_categories = {
            str(self.particle_category).strip().lower(),
            str(self.droplet_category).strip().lower(),
        }
        for raw_index, original in enumerate(raw_objects):
            obj = copy(original)
            obj["_raw_json_object_index"] = int(raw_index)
            objects.append(obj)
            category = str(obj.get("category", "")).strip().lower()
            if category not in selected_categories:
                continue
            points = np.asarray(obj.get("segmentation", []), dtype=np.float64)
            if (
                points.ndim == 2
                and points.shape[0] >= 3
                and points.shape[1] >= 2
                and np.all(np.isfinite(points[:, :2]))
            ):
                valid_points[raw_index] = points[:, :2]

        masks = self._rasterize_instance_masks(valid_points)
        kept = set(range(len(objects)))
        threshold = float(self.same_category_containment_threshold)

        # Same-category NMS: process largest masks first and suppress a smaller
        # mask when at least ``threshold`` of the smaller mask is already
        # covered.  This handles nested duplicates that have a modest IoU.
        for category in selected_categories:
            indices = [
                index
                for index in masks
                if str(objects[index].get("category", "")).strip().lower()
                == category
            ]
            indices.sort(key=lambda index: (-masks[index][1], index))
            accepted = []
            for index in indices:
                best = None
                for kept_index in accepted:
                    containment = self._mask_containment(
                        masks[index], masks[kept_index]
                    )
                    if best is None or containment > best[0]:
                        best = (containment, kept_index)
                if best is not None and best[0] >= threshold:
                    kept.discard(index)
                    self._record_suppressed_instance(
                        frame_name=frame_name,
                        removed=objects[index],
                        retained=objects[best[1]],
                        reason="same_category_containment",
                        overlap_fraction=best[0],
                    )
                    self.same_category_suppressed_count += 1
                else:
                    accepted.append(index)

        # A nanocluster prediction substantially contained in a nanodroplet is
        # a class-conflict duplicate.  Droplet masks take precedence because
        # the conflicting predictions in this dataset are particle fragments
        # inside a complete droplet mask.
        particle_category = str(self.particle_category).strip().lower()
        droplet_category = str(self.droplet_category).strip().lower()
        particle_indices = [
            index
            for index in masks
            if index in kept
            and str(objects[index].get("category", "")).strip().lower()
            == particle_category
        ]
        droplet_indices = [
            index
            for index in masks
            if index in kept
            and str(objects[index].get("category", "")).strip().lower()
            == droplet_category
        ]
        cross_threshold = float(self.particle_in_droplet_threshold)
        for particle_index in particle_indices:
            best = None
            particle_mask, particle_area = masks[particle_index]
            if particle_area <= 0:
                continue
            for droplet_index in droplet_indices:
                droplet_mask, _droplet_area = masks[droplet_index]
                coverage = float(np.count_nonzero(particle_mask & droplet_mask)) / float(
                    particle_area
                )
                if best is None or coverage > best[0]:
                    best = (coverage, droplet_index)
            if best is not None and best[0] >= cross_threshold:
                kept.discard(particle_index)
                self._record_suppressed_instance(
                    frame_name=frame_name,
                    removed=objects[particle_index],
                    retained=objects[best[1]],
                    reason="particle_contained_in_droplet",
                    overlap_fraction=best[0],
                )
                self.cross_category_suppressed_count += 1

        cleaned = dict(data)
        cleaned["objects"] = [obj for index, obj in enumerate(objects) if index in kept]
        if cache_key is not None:
            self._postprocessed_frame_cache[cache_key] = cleaned
        return cleaned

    @staticmethod
    def _rasterize_instance_masks(valid_points):
        if not valid_points:
            return {}

        all_points = np.vstack(list(valid_points.values()))
        left = int(floor(float(np.min(all_points[:, 0])))) - 1
        top = int(floor(float(np.min(all_points[:, 1])))) - 1
        right = int(ceil(float(np.max(all_points[:, 0])))) + 1
        bottom = int(ceil(float(np.max(all_points[:, 1])))) + 1
        width = max(1, right - left + 1)
        height = max(1, bottom - top + 1)

        masks = {}
        for index, points in valid_points.items():
            image = Image.new("1", (width, height), 0)
            polygon = [
                (float(point[0]) - left, float(point[1]) - top) for point in points
            ]
            ImageDraw.Draw(image).polygon(polygon, fill=1)
            mask = np.asarray(image, dtype=bool)
            masks[index] = (mask, int(np.count_nonzero(mask)))
        return masks

    @staticmethod
    def _mask_containment(first, second):
        first_mask, first_area = first
        second_mask, second_area = second
        smaller_area = min(int(first_area), int(second_area))
        if smaller_area <= 0:
            return 0.0
        intersection = int(np.count_nonzero(first_mask & second_mask))
        return float(intersection) / float(smaller_area)

    def _record_suppressed_instance(
        self, *, frame_name, removed, retained, reason, overlap_fraction
    ):
        self.instance_postprocess_records.append(
            {
                "frame_name": "" if frame_name is None else str(frame_name),
                "category": str(removed.get("category", "")),
                "removed_json_object_index": int(
                    removed.get("_raw_json_object_index", -1)
                ),
                "retained_category": str(retained.get("category", "")),
                "retained_json_object_index": int(
                    retained.get("_raw_json_object_index", -1)
                ),
                "reason": str(reason),
                "overlap_fraction": float(overlap_fraction),
            }
        )
