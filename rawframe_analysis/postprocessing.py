"""Frame-local and temporal cleanup for instance-segmentation masks."""

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
    """Merge strongly overlapping or substantially contacting same-class masks.

    Model predictions can contain a smaller mask almost completely inside a
    second mask for the same physical object.  Tracking such raw predictions
    creates false births and many short-lived IDs.  This mixin performs
    mask merging before any measurement, tracking, export, or
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
        overlap_threshold = float(self.same_category_containment_threshold)
        contact_gap_px = int(self.same_category_contact_gap_px)
        contact_threshold = float(self.same_category_contact_threshold)
        temporal_gap_px = int(self.same_category_temporal_gap_px)
        temporal_contact_threshold = float(
            self.same_category_temporal_contact_threshold
        )
        temporal_coverage_threshold = float(
            self.same_category_temporal_coverage_threshold
        )

        # Same-category union uses three complementary criteria:
        # 1. sufficient intersection relative to the smaller mask;
        # 2. a sufficiently long near-contact boundary within a small pixel gap.
        # 3. wider near-contact when both fragments belong to one prior mask.
        # The latter criteria handle one physical instance predicted as upper
        # and lower fragments whose masks have little or no intersection.
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

            temporal_relations = self._temporal_same_category_relations(
                valid_points=valid_points,
                indices=indices,
                category=category,
                current_masks=masks,
                gap_px=temporal_gap_px,
                contact_threshold=temporal_contact_threshold,
                coverage_threshold=temporal_coverage_threshold,
            )
            merge_relation_by_pair = {}
            for position, first in enumerate(indices):
                for second in indices[position + 1 :]:
                    overlap = self._mask_containment(masks[first], masks[second])
                    if overlap >= overlap_threshold:
                        merge_relation_by_pair[(first, second)] = (
                            "same_category_overlap_merge",
                            overlap,
                            0,
                        )
                        union(first, second)
                        continue
                    contact = self._mask_contact_fraction(
                        masks[first], masks[second], gap_px=contact_gap_px
                    )
                    if contact >= contact_threshold:
                        merge_relation_by_pair[(first, second)] = (
                            "same_category_contact_merge",
                            contact,
                            contact_gap_px,
                        )
                        union(first, second)
                        continue
                    temporal_relation = temporal_relations.get((first, second))
                    if temporal_relation is not None:
                        merge_relation_by_pair[(first, second)] = (
                            "same_category_temporal_contact_merge",
                            temporal_relation,
                            temporal_gap_px,
                        )
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
                member_set = set(members)
                bridge_gap_px = max(
                    (
                        relation_gap_px
                        for (first, second), (
                            _reason,
                            _score,
                            relation_gap_px,
                        ) in merge_relation_by_pair.items()
                        if relation_gap_px > 0
                        and first in member_set
                        and second in member_set
                    ),
                    default=0,
                )
                if bridge_gap_px > 0:
                    union_mask = self._bridge_nearby_masks(
                        union_mask, gap_px=bridge_gap_px
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
                    relations = [
                        relation
                        for pair, relation in merge_relation_by_pair.items()
                        if index in pair
                        and pair[0] in member_set
                        and pair[1] in member_set
                    ]
                    if relations:
                        reason, merge_score, _gap_px = max(
                            relations, key=lambda item: item[1]
                        )
                    else:
                        reason = "same_category_overlap_merge"
                        merge_score = max(
                            self._mask_containment(masks[index], masks[other])
                            for other in members
                            if other != index
                        )
                    self._record_suppressed_instance(
                        frame_name=frame_name,
                        removed=objects[index],
                        retained=objects[retained_index],
                        reason=reason,
                        overlap_fraction=merge_score,
                    )
                    self.same_category_suppressed_count += 1

        cleaned = dict(data)
        cleaned["objects"] = [
            obj for index, obj in enumerate(objects) if index in kept
        ]
        self._previous_postprocessed_objects = list(cleaned["objects"])
        if cache_key is not None:
            self._postprocessed_frame_cache[cache_key] = cleaned
        return cleaned

    def _temporal_same_category_relations(
        self,
        *,
        valid_points,
        indices,
        category,
        current_masks,
        gap_px,
        contact_threshold,
        coverage_threshold,
    ):
        """Find nearby current fragments covered by one prior same-class mask."""
        previous_objects = self._previous_postprocessed_objects or []
        previous_points = {}
        for previous_index, obj in enumerate(previous_objects):
            if str(obj.get("category", "")).strip().lower() != category:
                continue
            points = np.asarray(obj.get("segmentation", []), dtype=np.float64)
            if (
                points.ndim == 2
                and points.shape[0] >= 3
                and points.shape[1] >= 2
                and np.all(np.isfinite(points[:, :2]))
            ):
                previous_points[("previous", previous_index)] = points[:, :2]
        if not previous_points or len(indices) < 2:
            return {}

        joint_points = dict(previous_points)
        joint_points.update(
            {
                ("current", index): valid_points[index]
                for index in indices
                if index in valid_points
            }
        )
        joint_masks, _left, _top = self._rasterize_instance_masks(joint_points)
        relations = {}
        for position, first in enumerate(indices):
            for second in indices[position + 1 :]:
                contact = self._mask_contact_fraction(
                    current_masks[first], current_masks[second], gap_px=gap_px
                )
                if contact < contact_threshold:
                    continue
                first_key = ("current", first)
                second_key = ("current", second)
                shares_previous = any(
                    self._mask_coverage(joint_masks[first_key], previous_mask)
                    >= coverage_threshold
                    and self._mask_coverage(joint_masks[second_key], previous_mask)
                    >= coverage_threshold
                    for previous_key, previous_mask in joint_masks.items()
                    if previous_key[0] == "previous"
                )
                if shares_previous:
                    relations[(first, second)] = contact
        return relations

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
        if len(contours) == 1:
            contour = contours[0]
        else:
            # Contact-merged masks should normally be connected by closing.
            # The hull is a safe fallback for a subpixel raster gap that leaves
            # multiple exterior contours after closing.
            contour = cv2.convexHull(np.vstack(contours))
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

    @staticmethod
    def _mask_coverage(subject, reference):
        subject_mask, subject_area = subject
        if int(subject_area) <= 0:
            return 0.0
        intersection = int(np.count_nonzero(subject_mask & reference[0]))
        return float(intersection) / float(subject_area)

    @staticmethod
    def _mask_contact_fraction(first, second, gap_px):
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required to measure same-category mask contact"
            )
        first_mask = np.asarray(first[0], dtype=np.uint8)
        second_mask = np.asarray(second[0], dtype=np.uint8)
        edge_kernel = np.ones((3, 3), dtype=np.uint8)
        first_boundary = first_mask - cv2.erode(first_mask, edge_kernel)
        second_boundary = second_mask - cv2.erode(second_mask, edge_kernel)
        first_perimeter = int(np.count_nonzero(first_boundary))
        second_perimeter = int(np.count_nonzero(second_boundary))
        shorter_perimeter = min(first_perimeter, second_perimeter)
        if shorter_perimeter <= 0:
            return 0.0

        diameter = 2 * int(gap_px) + 1
        gap_kernel = np.ones((diameter, diameter), dtype=np.uint8)
        first_contact = int(
            np.count_nonzero(
                first_boundary & cv2.dilate(second_boundary, gap_kernel)
            )
        )
        second_contact = int(
            np.count_nonzero(
                second_boundary & cv2.dilate(first_boundary, gap_kernel)
            )
        )
        if first_perimeter <= second_perimeter:
            return float(first_contact) / float(first_perimeter)
        return float(second_contact) / float(second_perimeter)

    @staticmethod
    def _bridge_nearby_masks(mask, gap_px):
        if cv2 is None:
            raise RuntimeError("OpenCV is required to bridge nearby instance masks")
        diameter = 2 * int(gap_px) + 1
        kernel = np.ones((diameter, diameter), dtype=np.uint8)
        bridged = cv2.morphologyEx(
            np.asarray(mask, dtype=np.uint8), cv2.MORPH_CLOSE, kernel
        )
        return bridged.astype(bool)

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
