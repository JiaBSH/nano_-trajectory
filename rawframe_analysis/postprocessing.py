"""Frame-local cleanup for overlapping instance-segmentation masks."""

from __future__ import annotations

from copy import copy
from math import ceil, floor

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in incomplete environments
    cv2 = None


class InstancePostprocessingMixin:
    """Merge strongly overlapping masks that have the same category.

    Model predictions can contain a smaller mask almost completely inside a
    second mask for the same physical object.  Tracking such raw predictions
    creates false births and many short-lived IDs.  This mixin performs
    frame-local mask merging before any measurement, tracking, export, or
    drawing.  Different categories are deliberately independent: a particle
    and a droplet may occupy the same location and both remain valid.
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

        masks, mask_left, mask_top = self._rasterize_instance_masks(valid_points)
        kept = set(range(len(objects)))
        threshold = float(self.same_category_containment_threshold)

        # Same-category union: connect masks when their intersection covers at
        # least ``threshold`` of the smaller mask.  Connected groups are
        # replaced by their raster union, so protruding pixels from every
        # prediction remain part of the merged instance.
        for category in selected_categories:
            indices = [
                index
                for index in masks
                if str(objects[index].get("category", "")).strip().lower()
                == category
            ]
            parent = {index: index for index in indices}

            def find(index):
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            def union(first, second):
                first_root = find(first)
                second_root = find(second)
                if first_root != second_root:
                    parent[second_root] = first_root

            overlap_by_pair = {}
            for position, first in enumerate(indices):
                for second in indices[position + 1 :]:
                    overlap = self._mask_containment(masks[first], masks[second])
                    overlap_by_pair[(first, second)] = overlap
                    if overlap >= threshold:
                        union(first, second)

            groups = {}
            for index in indices:
                groups.setdefault(find(index), []).append(index)

            for members in groups.values():
                if len(members) < 2:
                    continue
                retained_index = min(members)
                union_mask = np.logical_or.reduce(
                    [masks[index][0] for index in members]
                )
                merged_segmentation = self._mask_union_polygon(
                    union_mask, mask_left, mask_top
                )
                if merged_segmentation is not None:
                    merged = copy(objects[retained_index])
                    merged["segmentation"] = merged_segmentation
                    merged["area"] = int(np.count_nonzero(union_mask))
                    objects[retained_index] = merged

                for index in members:
                    if index == retained_index:
                        continue
                    kept.discard(index)
                    pair = tuple(sorted((retained_index, index)))
                    overlap = overlap_by_pair.get(pair)
                    if overlap is None:
                        overlap = max(
                            self._mask_containment(masks[index], masks[other])
                            for other in members
                            if other != index
                        )
                    self._record_suppressed_instance(
                        frame_name=frame_name,
                        removed=objects[index],
                        retained=objects[retained_index],
                        reason="same_category_overlap_merge",
                        overlap_fraction=overlap,
                    )
                    self.same_category_suppressed_count += 1

        cleaned = dict(data)
        cleaned["objects"] = [obj for index, obj in enumerate(objects) if index in kept]
        if cache_key is not None:
            self._postprocessed_frame_cache[cache_key] = cleaned
        return cleaned

    @staticmethod
    def _rasterize_instance_masks(valid_points):
        if not valid_points:
            return {}, 0, 0

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
        return masks, left, top

    @staticmethod
    def _mask_union_polygon(mask, left, top):
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required to merge overlapping instance masks"
            )
        contours, _hierarchy = cv2.findContours(
            np.asarray(mask, dtype=np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        points = contour[:, 0, :].astype(np.float64)
        if points.shape[0] < 3:
            return None
        points[:, 0] += float(left)
        points[:, 1] += float(top)
        return points.tolist()

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
