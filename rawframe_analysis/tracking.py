"""Cross-frame object association and derived time-series helpers."""

from __future__ import annotations

import numpy as np


class ObjectTrackingMixin:
    """Provide cross-frame object association and derived time-series helpers."""

    def _object_detections_by_frame(self):
        if self._object_detections_by_frame_cache is not None:
            return self._object_detections_by_frame_cache

        from collections import defaultdict

        by_frame = defaultdict(list)
        for (
            frame_id,
            frame_name,
            nm_per_px,
            cx_nm,
            cy_nm,
            area_nm2,
        ) in self.object_records:
            by_frame[int(frame_id)].append(
                (
                    frame_name,
                    float(nm_per_px),
                    float(cx_nm),
                    float(cy_nm),
                    float(area_nm2),
                )
            )

        self._object_detections_by_frame_cache = by_frame
        return by_frame

    def _event_id_series_for_object_records(self, max_dist=50.0):
        key = (len(self.object_records), float(max_dist))
        cached = self._event_id_series_cache.get(key)
        if cached is not None:
            return cached

        result = self._build_event_id_series_with_assignments(
            self._object_detections_by_frame(),
            max_dist=max_dist,
        )
        self._event_id_series_cache[key] = result
        return result

    @staticmethod
    def _compute_speed_series_from_points(points, frame_interval_s=1.0):
        """Compute speed series from a list of points.

        points: [(frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2), ...]
        returns: [(frame_id, frame_name, speed_nm_per_s), ...] aligned to the *current* frame.
        """
        if not points:
            return []

        dt = float(frame_interval_s)
        if dt <= 0:
            raise ValueError(f"frame_interval_s must be > 0, got {frame_interval_s}")

        # sort by frame
        pts = sorted(points, key=lambda p: int(p[0]))
        out = []
        prev = pts[0]
        for cur in pts[1:]:
            f0, _name0, _nm0, x0, y0, _a0 = prev
            f1, name1, _nm1, x1, y1, _a1 = cur
            df = int(f1) - int(f0)
            if df <= 0:
                prev = cur
                continue
            dist = float(np.hypot(float(x1) - float(x0), float(y1) - float(y0)))
            out.append((int(f1), str(name1), dist / (float(df) * dt)))
            prev = cur
        return out

    @staticmethod
    def _bin_speed_series(speed_points, bin_size_frames=10):
        """Bin speed series into non-overlapping frame windows and take the mean.

        speed_points: [(frame_id, frame_name, speed_nm_per_s), ...]
        returns: [(frame_id, frame_name, mean_speed_nm_per_s), ...]
                 where frame_id/frame_name correspond to the last point in that bin.
        """
        if not speed_points:
            return []
        b = int(bin_size_frames)
        if b <= 0:
            raise ValueError(f"bin_size_frames must be > 0, got {bin_size_frames}")

        from collections import defaultdict

        buckets = defaultdict(
            list
        )  # bin_index -> list of (frame_id, frame_name, speed)
        for frame_id, frame_name, speed in speed_points:
            idx = int(frame_id) // b
            buckets[idx].append((int(frame_id), str(frame_name), float(speed)))

        out = []
        for idx in sorted(buckets.keys()):
            items = sorted(buckets[idx], key=lambda t: t[0])
            if not items:
                continue
            frame_last, name_last, _ = items[-1]
            mean_speed = float(np.mean([s for _f, _n, s in items]))
            out.append((int(frame_last), str(name_last), mean_speed))

        return out

    @staticmethod
    def _display_id_mapping(series_by_id):
        """Map internal instance_id -> display_id (1..K) by first appearance."""
        instance_ids = sorted([int(k) for k in series_by_id.keys()])
        first_frame_by_id = {}
        for iid in instance_ids:
            pts = series_by_id.get(iid) or []
            if len(pts) == 0:
                continue
            first_frame_by_id[iid] = int(min(p[0] for p in pts))

        ordered_ids = sorted(
            first_frame_by_id.keys(), key=lambda i: (first_frame_by_id[i], i)
        )
        return {iid: idx + 1 for idx, iid in enumerate(ordered_ids)}

    @staticmethod
    def _select_plot_instance_ids(series_by_id, max_plot_tracks=None):
        """Return track ids to draw, preferring longer tracks when the plot is crowded."""
        ids = [int(k) for k, pts in series_by_id.items() if len(pts) > 0]
        total = len(ids)
        if (
            max_plot_tracks is None
            or int(max_plot_tracks) <= 0
            or total <= int(max_plot_tracks)
        ):
            return sorted(ids), total

        def sort_key(iid):
            pts = series_by_id.get(iid) or []
            first_frame = min((int(p[0]) for p in pts), default=10**12)
            return (-len(pts), first_frame, int(iid))

        selected = sorted(ids, key=sort_key)[: int(max_plot_tracks)]
        return selected, total

    @staticmethod
    def _positive_plot_limit(value):
        if value is None:
            return None
        value = int(value)
        return value if value > 0 else None

    def _build_event_id_series(
        self, detections_by_frame, max_dist=50.0, return_assignments=False
    ):
        """Assign globally-incrementing ids with continuity-first linking.

        Rules:
        - First frame detections get ids 1..N
        - Consecutive frames are linked by nearest-neighbor (one-to-one) within max_dist
        - Matched detections keep previous ids, even if object counts change
        - Unmatched current detections get NEW ids

                detections_by_frame: dict[int, list[tuple[frame_name,nm_per_px,cx_nm,cy_nm,area_nm2]]]
                returns:
                    - if return_assignments=False: (series_by_id, events)
                    - if return_assignments=True: (series_by_id, assigned_ids_by_frame, events)
        """
        from collections import defaultdict

        frames_sorted = sorted(detections_by_frame.keys())
        if not frames_sorted:
            if bool(return_assignments):
                return {}, {}, []
            return {}, []

        next_id = 1
        assigned_ids_by_frame = {}
        events = []

        # init: first frame
        f0 = frames_sorted[0]
        det0 = detections_by_frame[f0]
        ids0 = []
        for _ in det0:
            ids0.append(next_id)
            next_id += 1
        assigned_ids_by_frame[f0] = ids0

        prev_dets = det0
        prev_ids = ids0

        for frame in frames_sorted[1:]:
            curr_dets = detections_by_frame[frame]
            n_prev = len(prev_dets)
            n_curr = len(curr_dets)
            curr_ids = [None] * n_curr

            if n_prev == 0:
                for j in range(n_curr):
                    curr_ids[j] = next_id
                    events.append(
                        {"frame": frame, "type": "birth", "dst_id": int(next_id)}
                    )
                    next_id += 1
                assigned_ids_by_frame[frame] = curr_ids
                prev_frame, prev_dets, prev_ids = frame, curr_dets, curr_ids
                continue

            if n_curr == 0:
                assigned_ids_by_frame[frame] = []
                prev_dets, prev_ids = curr_dets, []
                continue

            # Do one-to-one assignment by minimal distance for all count combinations.
            prev_xy = np.array([[d[2], d[3]] for d in prev_dets], dtype=np.float64)
            curr_xy = np.array([[d[2], d[3]] for d in curr_dets], dtype=np.float64)
            dists = np.linalg.norm(prev_xy[:, None, :] - curr_xy[None, :, :], axis=2)

            pairs = []  # (dist, i_prev, j_curr)
            for i in range(n_prev):
                for j in range(n_curr):
                    dist = float(dists[i, j])
                    if dist <= float(max_dist):
                        pairs.append((dist, i, j))
            pairs.sort(key=lambda x: x[0])

            used_prev = set()
            used_curr = set()
            for _dist, i, j in pairs:
                if i in used_prev or j in used_curr:
                    continue
                curr_ids[j] = int(prev_ids[i])
                used_prev.add(i)
                used_curr.add(j)

            # any unmatched current object becomes a new id
            for j in range(n_curr):
                if curr_ids[j] is None:
                    curr_ids[j] = int(next_id)
                    events.append(
                        {"frame": frame, "type": "birth", "dst_id": int(next_id)}
                    )
                    next_id += 1

            assigned_ids_by_frame[frame] = curr_ids
            prev_dets, prev_ids = curr_dets, curr_ids

        # build series
        series_by_id = defaultdict(list)
        for frame in frames_sorted:
            dets = detections_by_frame[frame]
            ids = assigned_ids_by_frame.get(frame, [])
            for det, instance_id in zip(dets, ids):
                frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 = det
                series_by_id[int(instance_id)].append(
                    (
                        int(frame),
                        frame_name,
                        float(nm_per_px),
                        float(cx_nm),
                        float(cy_nm),
                        float(area_nm2),
                    )
                )

        if bool(return_assignments):
            return dict(series_by_id), assigned_ids_by_frame, events
        return dict(series_by_id), events

    def _build_event_id_series_with_assignments(
        self, detections_by_frame, max_dist=50.0
    ):
        """Compatibility helper: return series + per-frame assignment list + events."""
        return self._build_event_id_series(
            detections_by_frame, max_dist=max_dist, return_assignments=True
        )

    def _build_greedy_tracks(self, detections_by_frame, max_dist=50.0):
        """Greedy link detections in consecutive frames into tracks.

        detections_by_frame: dict[int, list[tuple[frame_name,nm_per_px,cx_nm,cy_nm,area_nm2]]]
        max_dist: distance threshold in nm
        returns: list of tracks, each track: {'last_frame': int, 'points': [(frame_id,frame_name,nm_per_px,cx_nm,cy_nm,area_nm2), ...]}
        """
        tracks = []
        for frame in sorted(detections_by_frame.keys()):
            dets = detections_by_frame[frame]
            assigned = [False] * len(dets)

            # extend tracks from previous frame
            for t in tracks:
                if t["last_frame"] != frame - 1:
                    continue

                last_x, last_y = t["points"][-1][3], t["points"][-1][4]
                best_idx = None
                best_dist = float("inf")
                for i, (frame_name, nm_per_px, cx_nm, cy_nm, area_nm2) in enumerate(
                    dets
                ):
                    if assigned[i]:
                        continue
                    d = np.hypot(cx_nm - last_x, cy_nm - last_y)
                    if d < best_dist:
                        best_dist = d
                        best_idx = i

                if best_idx is not None and best_dist <= max_dist:
                    frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 = dets[best_idx]
                    t["points"].append(
                        (frame, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2)
                    )
                    t["last_frame"] = frame
                    assigned[best_idx] = True

            # create new tracks for unassigned detections
            for i, (frame_name, nm_per_px, cx_nm, cy_nm, area_nm2) in enumerate(dets):
                if not assigned[i]:
                    tracks.append(
                        {
                            "last_frame": frame,
                            "points": [
                                (frame, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2)
                            ],
                        }
                    )

        return tracks
