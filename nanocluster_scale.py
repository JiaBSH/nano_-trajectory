import os
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import Normalize
from matplotlib import font_manager as fm


class GasTracker:
    def __init__(
        self,
        json_dir,
        image_path,
        scale_csv=None,
        scale_value_nm=20.0,
        strict_scale_match=False,
        gas_category="gas",
        pin_category="pin"
    ):
        self.json_dir = json_dir
        self.image_path = image_path
        self.scale_csv = scale_csv
        self.scale_value_nm = float(scale_value_nm)
        self.strict_scale_match = bool(strict_scale_match)
        self.gas_category = gas_category
        self.pin_category = pin_category

        self.json_files = self._load_and_sort_jsons()

        # per-frame scale map: {frame_stem: nm_per_pixel}
        self.scale_map = {}
        self.fallback_nm_per_px = None
        self.max_nm_per_px = None
        self.min_nm_per_px = None
        if self.scale_csv is not None:
            self.scale_map = self._load_nm_per_px_map(self.scale_csv, default_scale_value_nm=self.scale_value_nm)
            if len(self.scale_map) > 0:
                vals = np.array(list(self.scale_map.values()), dtype=np.float64)
                self.fallback_nm_per_px = float(np.median(vals))
                self.max_nm_per_px = float(np.max(vals))
                self.min_nm_per_px = float(np.min(vals))
            else:
                raise ValueError(f"Scale CSV provided but no usable rows found: {self.scale_csv}")

        # 数据容器（全部使用真实尺寸：nm / nm^2）
        self.area_records = []        # [frame_id, frame_name, nm_per_px, area_nm2]
        self.contour_records = []     # [frame_id, frame_name, "(x_nm,y_nm)", ...]
        self.centroid_records = []    # [frame_id, frame_name, nm_per_px, cx_nm, cy_nm]
        self.object_records = []      # [frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2]

        # pin 参考
        self.ref_pin_centroid = None
        self.last_shift = np.zeros(2)

        # 画图准备
        img = Image.open(image_path)
        self.W, self.H = img.size

        # Make sure Chinese text can render on Windows (avoid "□□□" tofu boxes)
        self._configure_matplotlib_fonts()

    @staticmethod
    def _configure_matplotlib_fonts():
        """Configure Matplotlib fonts for Chinese text.

        If suitable CJK fonts aren't available, Matplotlib will fall back and may show tofu boxes.
        """
        preferred = [
            "Microsoft YaHei",  # 微软雅黑
            "SimHei",           # 黑体
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

        plt.rcParams["axes.unicode_minus"] = False

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
    def _load_nm_per_px_map(cls, csv_path, default_scale_value_nm=20.0):
        """Load per-image nm/px from a scalebar CSV.

        Supports:
        - minimal CSV: image,pixel_length
        - yolo_easyocr output: image,scale_value,unit,pixel_length,ratio,...

        Keying:
        - uses image basename stem, e.g. '..._000000000003'
        """
        csv_path = str(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Scale CSV not found: {csv_path}. "
                "Please provide a CSV with columns 'image' and 'pixel_length'."
            )

        nm_per_px = {}
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
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
                nm_per_px[stem] = float(scale_nm) / float(pixel_length)

        return nm_per_px

    def _nm_per_px_for_frame(self, frame_name):
        if self.scale_csv is None:
            raise ValueError(
                "scale_csv is required to output real units. "
                "Provide the scalebar CSV (columns: image,pixel_length)."
            )
        v = self.scale_map.get(frame_name)
        if v is not None:
            return float(v)
        if self.strict_scale_match:
            raise KeyError(f"No scale entry for frame '{frame_name}' in {self.scale_csv}")
        # user requested: if no matching scale for this frame, skip it
        return None

    # -----------------------------
    # 工具函数
    # -----------------------------
    def _load_and_sort_jsons(self):
        files = [
            f for f in os.listdir(self.json_dir)
            if f.endswith(".json")
        ]
        files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        return files

    @staticmethod
    def polygon_area(coords):
        """
        coords: (N,2) 不需要闭合
        """
        x = coords[:, 0]
        y = coords[:, 1]
        return 0.5 * abs(
            np.dot(x, np.roll(y, -1)) -
            np.dot(y, np.roll(x, -1))
        )

    # -----------------------------
    # 主处理流程
    # -----------------------------
    def process_all_frames(self):
        for frame_id, json_name in enumerate(self.json_files):
            json_path = os.path.join(self.json_dir, json_name)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            frame_name = Path(json_name).stem
            try:
                nm_per_px = self._nm_per_px_for_frame(frame_name)
            except Exception as e:
                print(f"[skip] frame_id={frame_id} frame_name={frame_name}: scale lookup error: {e}")
                continue
            if nm_per_px is None:
                print(f"[skip] frame_id={frame_id} frame_name={frame_name}: no matching scale in CSV")
                continue

            shift = self._compute_pin_shift(data)

            self._process_gas_objects(
                data,
                frame_id,
                frame_name,
                nm_per_px,
                shift
            )

    def _compute_pin_shift(self, data):
        pin_pts = []
        for obj in data.get("objects", []):
            if obj.get("category") == self.pin_category:
                pin_pts.append(
                    np.array(obj["segmentation"], dtype=np.float32)
                )

        if len(pin_pts) > 0:
            pin_pts = np.vstack(pin_pts)
            pin_centroid = pin_pts.mean(axis=0)

            if self.ref_pin_centroid is None:
                self.ref_pin_centroid = pin_centroid.copy()

            shift = pin_centroid - self.ref_pin_centroid
            self.last_shift = shift
        else:
            shift = self.last_shift

        return shift

    def _process_gas_objects(self, data, frame_id, frame_name, nm_per_px, shift):
        for obj in data.get("objects", []):
            if obj.get("category") != self.gas_category:
                continue

            pts = np.array(obj["segmentation"], dtype=np.float32)
            pts = pts - shift   # ★ 去整体漂移

            if pts.shape[0] < 3:
                continue

            # ---- 面积 ----
            area_px2 = self.polygon_area(pts)
            area_nm2 = float(area_px2) * float(nm_per_px) * float(nm_per_px)
            self.area_records.append([frame_id, frame_name, float(nm_per_px), area_nm2])

            # ---- 质心 ----
            centroid = pts.mean(axis=0)
            cx_px, cy_px = float(centroid[0]), float(centroid[1])
            cx_nm, cy_nm = cx_px * float(nm_per_px), cy_px * float(nm_per_px)
            self.centroid_records.append([frame_id, frame_name, float(nm_per_px), cx_nm, cy_nm])

            # ---- 每个目标的聚合记录（用于追踪面积曲线）----
            self.object_records.append([frame_id, frame_name, float(nm_per_px), cx_nm, cy_nm, area_nm2])

            # ---- 轮廓（每帧一行）----
            row = [frame_id, frame_name]
            for (x, y) in pts:
                x_nm = float(x) * float(nm_per_px)
                y_nm = float(y) * float(nm_per_px)
                row.append(f"({x_nm:.3f},{y_nm:.3f})")
            self.contour_records.append(row)

    # -----------------------------
    # 数据导出
    # -----------------------------
    def export_results(self):
        # 面积
        with open(f"{self.gas_category}_area_vs_frame.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "frame_name", "nm_per_pixel", "area_nm2"])
            writer.writerows(
                [[frame_id, frame_name, f"{nm_per_px:.6f}", f"{area_nm2:.6f}"]
                 for frame_id, frame_name, nm_per_px, area_nm2 in self.area_records]
            )

        # 轮廓（每帧一行）
        with open(f"{self.gas_category}_contours_by_frame.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "frame_name", "contour_points_nm"])
            writer.writerows(self.contour_records)

        # 质心
        with open(f"{self.gas_category}_centroids.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "frame_name", "nm_per_pixel", "cx_nm", "cy_nm"])
            writer.writerows(
                [[frame_id, frame_name, f"{nm_per_px:.6f}", f"{cx_nm:.6f}", f"{cy_nm:.6f}"]
                 for frame_id, frame_name, nm_per_px, cx_nm, cy_nm in self.centroid_records]
            )

        print("Export finished:")
        print(f" - {self.gas_category}_area_vs_frame.csv")
        print(f" - {self.gas_category}_contours_by_frame.csv")

    def export_tracked_area_results(self, tracks, out_csv=None):
        """Export tracked area series.

        CSV columns: track_id, frame_id, frame_name, nm_per_pixel, area_nm2, cx_nm, cy_nm
        """
        if out_csv is None:
            out_csv = f"{self.gas_category}_tracked_area_vs_frame.csv"

        rows = []
        for track_id, t in enumerate(tracks):
            for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in t['points']:
                rows.append(
                    [track_id, frame_id, frame_name, f"{nm_per_px:.6f}", f"{area_nm2:.6f}", f"{cx_nm:.6f}", f"{cy_nm:.6f}"]
                )

        rows.sort(key=lambda r: (r[0], r[1]))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "frame_id", "frame_name", "nm_per_pixel", "area_nm2", "cx_nm", "cy_nm"])
            writer.writerows(rows)

        print(f" - {out_csv}")

    def export_id_series(self, series_by_id, out_csv=None):
        """Export area series keyed by a globally-incrementing instance id.

        CSV columns: instance_id, frame_id, frame_name, nm_per_pixel, area_nm2, cx_nm, cy_nm
        """
        if out_csv is None:
            out_csv = f"{self.gas_category}_instance_area_vs_frame.csv"

        rows = []
        for instance_id, points in series_by_id.items():
            for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in points:
                rows.append(
                    [
                        int(instance_id),
                        int(frame_id),
                        frame_name,
                        f"{float(nm_per_px):.6f}",
                        f"{float(area_nm2):.6f}",
                        f"{float(cx_nm):.6f}",
                        f"{float(cy_nm):.6f}",
                    ]
                )

        rows.sort(key=lambda r: (r[0], r[1]))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "frame_id", "frame_name", "nm_per_pixel", "area_nm2", "cx_nm", "cy_nm"])
            writer.writerows(rows)

        print(f" - {out_csv}")

    def export_speed_series(self, speed_series_by_id, out_csv=None):
        """Export per-instance speed series (from centroid displacement).

        Speed is computed between consecutive detections of the same instance:
            speed = distance_nm / (delta_frame * frame_interval_s)

        CSV columns: instance_id, frame_id, frame_name, speed_nm_per_s
        """
        if out_csv is None:
            out_csv = f"{self.gas_category}_instance_speed_vs_frame.csv"

        rows = []
        for instance_id, points in speed_series_by_id.items():
            for frame_id, frame_name, speed_nm_per_s in points:
                rows.append([int(instance_id), int(frame_id), frame_name, f"{float(speed_nm_per_s):.6f}"])

        rows.sort(key=lambda r: (r[0], r[1]))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "frame_id", "frame_name", "speed_nm_per_s"])
            writer.writerows(rows)

        print(f" - {out_csv}")

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

        buckets = defaultdict(list)  # bin_index -> list of (frame_id, frame_name, speed)
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

        ordered_ids = sorted(first_frame_by_id.keys(), key=lambda i: (first_frame_by_id[i], i))
        return {iid: idx + 1 for idx, iid in enumerate(ordered_ids)}

    def _build_event_id_series(self, detections_by_frame, max_dist=50.0):
        """Assign globally-incrementing ids with merge/split relabeling.

        Rules:
        - First frame detections get ids 1..N
        - Merge (many prev -> one curr): curr gets a NEW id
        - Split (one prev -> many curr): each child gets a NEW id
        - 1-to-1 continuation keeps the same id

        detections_by_frame: dict[int, list[tuple[frame_name,nm_per_px,cx_nm,cy_nm,area_nm2]]]
        returns: (series_by_id, events)
        """
        from collections import defaultdict

        frames_sorted = sorted(detections_by_frame.keys())
        if not frames_sorted:
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
                    events.append({"frame": frame, "type": "birth", "dst_id": int(next_id)})
                    next_id += 1
                assigned_ids_by_frame[frame] = curr_ids
                prev_frame, prev_dets, prev_ids = frame, curr_dets, curr_ids
                continue

            if n_curr == 0:
                assigned_ids_by_frame[frame] = []
                prev_dets, prev_ids = curr_dets, []
                continue

            # If object count changes, treat as merge/split and relabel ALL current objects.
            # This avoids false merge/split when objects are merely close.
            if n_prev != n_curr:
                new_ids = []
                for j in range(n_curr):
                    curr_ids[j] = int(next_id)
                    new_ids.append(int(next_id))
                    next_id += 1

                if n_curr < n_prev:
                    events.append(
                        {
                            "frame": frame,
                            "type": "merge",
                            "src_ids": [int(x) for x in prev_ids],
                            "dst_ids": [int(x) for x in new_ids],
                        }
                    )
                else:
                    events.append(
                        {
                            "frame": frame,
                            "type": "split",
                            "src_ids": [int(x) for x in prev_ids],
                            "dst_ids": [int(x) for x in new_ids],
                        }
                    )

                assigned_ids_by_frame[frame] = curr_ids
                prev_dets, prev_ids = curr_dets, curr_ids
                continue

            # n_prev == n_curr: do a one-to-one assignment by minimal distance.
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
                    events.append({"frame": frame, "type": "birth", "dst_id": int(next_id)})
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
                series_by_id[int(instance_id)].append((int(frame), frame_name, float(nm_per_px), float(cx_nm), float(cy_nm), float(area_nm2)))

        return dict(series_by_id), events

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
                if t['last_frame'] != frame - 1:
                    continue

                last_x, last_y = t['points'][-1][3], t['points'][-1][4]
                best_idx = None
                best_dist = float('inf')
                for i, (frame_name, nm_per_px, cx_nm, cy_nm, area_nm2) in enumerate(dets):
                    if assigned[i]:
                        continue
                    d = np.hypot(cx_nm - last_x, cy_nm - last_y)
                    if d < best_dist:
                        best_dist = d
                        best_idx = i

                if best_idx is not None and best_dist <= max_dist:
                    frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 = dets[best_idx]
                    t['points'].append((frame, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2))
                    t['last_frame'] = frame
                    assigned[best_idx] = True

            # create new tracks for unassigned detections
            for i, (frame_name, nm_per_px, cx_nm, cy_nm, area_nm2) in enumerate(dets):
                if not assigned[i]:
                    tracks.append({'last_frame': frame, 'points': [(frame, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2)]})

        return tracks

    def plot_area_trajectories(self, max_dist=50.0, min_track_length=1, outname=None, id_mode="event"):
        """Plot each droplet's area-vs-frame curve in one figure.

        Tracks are built by greedy centroid linking.
        NOTE: max_dist is in nm because centroids are stored in nm.
        """
        if len(self.object_records) == 0:
            print("No object records to plot area trajectories.")
            return

        from collections import defaultdict

        by_frame = defaultdict(list)
        for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in self.object_records:
            by_frame[int(frame_id)].append((frame_name, float(nm_per_px), float(cx_nm), float(cy_nm), float(area_nm2)))

        if str(id_mode).lower() == "greedy":
            tracks = self._build_greedy_tracks(by_frame, max_dist=max_dist)
            tracks = [t for t in tracks if len(t['points']) >= int(min_track_length)]
            series_by_id = {int(track_id): [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in t["points"]] for track_id, t in enumerate(tracks)}
        else:
            series_by_id, _events = self._build_event_id_series(by_frame, max_dist=max_dist)
            series_by_id = {k: v for k, v in series_by_id.items() if len(v) >= int(min_track_length)}

        if outname is None:
            outname = f"{self.gas_category}_area_trajectories.png"

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlabel("Frame id")
        ax.set_ylabel("Area (nm^2)")
        ax.grid(True, alpha=0.25)

        # color cycle (good for dozens of tracks; for hundreds, they'll repeat)
        cmap = plt.cm.tab20

        display_id_of = self._display_id_mapping(series_by_id)
        instance_ids = sorted(series_by_id.keys())

        line_handles = []
        line_labels = []

        for instance_id in instance_ids:
            pts = series_by_id[instance_id]
            frames = np.array([p[0] for p in pts], dtype=np.int32)
            areas = np.array([p[5] for p in pts], dtype=np.float32)
            if frames.size == 0:
                continue
            order = np.argsort(frames)
            frames = frames[order]
            areas = areas[order]

            disp_id = display_id_of.get(int(instance_id), 0)
            color = cmap(int(disp_id) % 20)
            (line,) = ax.plot(frames, areas, color=color, linewidth=1.2, alpha=0.85)
            line_handles.append(line)
            line_labels.append(str(int(disp_id) if disp_id > 0 else int(instance_id)))

        # 图例：自适应布局，避免挡线 & 避免图被挤得很“扁”
        if len(line_handles) > 0:
            n_items = len(line_handles)

            # Prefer right-side legend for moderate counts; switch to multi-column / bottom for very long legends.
            if n_items <= 20:
                ncol = 1
                fig.set_size_inches(12, 6, forward=True)
                fig.subplots_adjust(right=0.80)
                ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=8,
                    ncol=ncol,
                )
            elif n_items <= 60:
                ncol = 2
                fig.set_size_inches(14, 6, forward=True)
                fig.subplots_adjust(right=0.78)
                ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=7,
                    ncol=ncol,
                    columnspacing=0.8,
                    handlelength=1.2,
                )
            else:
                # Too many: put legend below to keep plot area wide.
                ncol = min(6, int(np.ceil(n_items / 20.0)))
                fig.set_size_inches(14, 8, forward=True)
                fig.subplots_adjust(bottom=0.22)
                ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.12),
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=7,
                    ncol=ncol,
                    columnspacing=0.8,
                    handlelength=1.2,
                )

        ax.set_title(f"{self.gas_category}: area vs frame (per droplet track)", loc="center")
        plt.tight_layout()
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        print(f"Saved area trajectories plot: {outname}")

        # also export tracked series for downstream analysis
        if str(id_mode).lower() == "greedy":
            self.export_tracked_area_results(tracks)
        else:
            self.export_id_series(series_by_id)

    def plot_velocity_trajectories(
        self,
        max_dist=50.0,
        min_track_length=1,
        outname=None,
        id_mode="event",
        frame_interval_s=1.0,
        bin_size_frames=10,
    ):
        """Plot each individual's speed-vs-frame curve.

        Speed is computed from centroid displacement between consecutive detections.
        NOTE: speed unit is nm/s; set frame_interval_s (seconds per frame) to match your acquisition.
        """
        if len(self.object_records) == 0:
            print("No object records to plot velocity trajectories.")
            return

        from collections import defaultdict

        by_frame = defaultdict(list)
        for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in self.object_records:
            by_frame[int(frame_id)].append((frame_name, float(nm_per_px), float(cx_nm), float(cy_nm), float(area_nm2)))

        if str(id_mode).lower() == "greedy":
            tracks = self._build_greedy_tracks(by_frame, max_dist=max_dist)
            tracks = [t for t in tracks if len(t["points"]) >= int(min_track_length)]
            series_by_id = {
                int(track_id): [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in t["points"]]
                for track_id, t in enumerate(tracks)
            }
        else:
            series_by_id, _events = self._build_event_id_series(by_frame, max_dist=max_dist)
            series_by_id = {k: v for k, v in series_by_id.items() if len(v) >= int(min_track_length)}

        # compute speed series for each id (raw, per-frame)
        speed_series_by_id = {}
        for instance_id, pts in series_by_id.items():
            sp = self._compute_speed_series_from_points(pts, frame_interval_s=frame_interval_s)
            if len(sp) > 0:
                speed_series_by_id[int(instance_id)] = sp

        # bin-average: every N frames a mean value
        b = int(bin_size_frames)
        if b <= 0:
            raise ValueError(f"bin_size_frames must be > 0, got {bin_size_frames}")

        if b == 1:
            binned_speed_by_id = dict(speed_series_by_id)
        else:
            binned_speed_by_id = {}
            for instance_id, sp in speed_series_by_id.items():
                bp = self._bin_speed_series(sp, bin_size_frames=b)
                if len(bp) > 0:
                    binned_speed_by_id[int(instance_id)] = bp

        if outname is None:
            if b == 1:
                outname = f"{self.gas_category}_velocity_trajectories.png"
            else:
                outname = f"{self.gas_category}_velocity_mean_{b}frames.png"

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlabel("Frame id")
        ax.set_ylabel("Speed (nm/s)")
        ax.grid(True, alpha=0.25)

        cmap = plt.cm.tab20
        display_id_of = self._display_id_mapping(binned_speed_by_id)
        instance_ids = sorted(binned_speed_by_id.keys())

        line_handles = []
        line_labels = []

        for instance_id in instance_ids:
            pts = binned_speed_by_id[instance_id]
            frames = np.array([p[0] for p in pts], dtype=np.int32)
            speeds = np.array([p[2] for p in pts], dtype=np.float32)
            if frames.size == 0:
                continue
            order = np.argsort(frames)
            frames = frames[order]
            speeds = speeds[order]

            disp_id = display_id_of.get(int(instance_id), 0)
            color = cmap(int(disp_id) % 20)
            (line,) = ax.plot(frames, speeds, color=color, linewidth=1.2, alpha=0.85)
            line_handles.append(line)
            line_labels.append(str(int(disp_id) if disp_id > 0 else int(instance_id)))

        # legend layout (same idea as area plot)
        if len(line_handles) > 0:
            n_items = len(line_handles)
            if n_items <= 20:
                fig.set_size_inches(12, 6, forward=True)
                fig.subplots_adjust(right=0.80)
                ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=8,
                    ncol=1,
                )
            elif n_items <= 60:
                fig.set_size_inches(14, 6, forward=True)
                fig.subplots_adjust(right=0.78)
                ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=7,
                    ncol=2,
                    columnspacing=0.8,
                    handlelength=1.2,
                )
            else:
                ncol = min(6, int(np.ceil(n_items / 20.0)))
                fig.set_size_inches(14, 8, forward=True)
                fig.subplots_adjust(bottom=0.22)
                ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.12),
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=7,
                    ncol=ncol,
                    columnspacing=0.8,
                    handlelength=1.2,
                )

        if b == 1:
            ax.set_title(f"{self.gas_category}: velocity vs frame (per track)", loc="center")
        else:
            ax.set_title(f"{self.gas_category}: mean velocity per {b} frames", loc="center")
        plt.tight_layout()
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        print(f"Saved velocity trajectories plot: {outname}")

        # export for downstream analysis
        if b == 1:
            self.export_speed_series(speed_series_by_id)
        else:
            self.export_speed_series(
                binned_speed_by_id,
                out_csv=f"{self.gas_category}_instance_speed_mean_{b}frames.csv",
            )

    # Alias for naming preference
    def plot_speed_trajectories(self, *args, **kwargs):
        return self.plot_velocity_trajectories(*args, **kwargs)
    # -----------------------------
    # 可视化（抽帧）
    # -----------------------------
    def plot_evolution(self, step=200):
        fig, ax = plt.subplots(figsize=(8, 8))
        scale = float(self.max_nm_per_px) if self.max_nm_per_px is not None else 1.0
        ax.set_xlim(0, self.W * scale * 1.5)
        ax.set_ylim(self.H * scale, 0)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_aspect("equal", adjustable="box")

        cmap = plt.cm.plasma
        norm = Normalize(vmin=0, vmax=len(self.json_files) - 1)

        for row in self.contour_records:
            frame_id = row[0]
            if frame_id % step != 0:
                continue

            pts = []
            # row format: [frame_id, frame_name, "(x_nm,y_nm)", ...]
            for item in row[2:]:
                x, y = map(
                    float,
                    item.strip("()").split(",")
                )
                pts.append([x, y])

            pts = np.array(pts)
            pts = np.vstack([pts, pts[0]])

            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=cmap(norm(frame_id)),
                linewidth=1.5,
                alpha=0.85
            )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Frame id colorbar: same height as the axes
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.10)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Frame id")

        ax.set_title(f"{self.gas_category} domain evolution (pin-referenced)", loc="center")
        plt.tight_layout()
        # add a visible border around the axes
        from matplotlib.patches import Rectangle
        border_width = 3
        border_color = "black"
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                 fill=False, edgecolor=border_color,
                 linewidth=border_width, zorder=10, clip_on=False)
        ax.add_patch(rect)
        plt.savefig(f"{self.gas_category}_evolution.png", dpi=300, bbox_inches="tight")

    def plot_centroid_trajectories(self, max_dist=50.0):
        """
        Build simple greedy tracks by linking centroids in consecutive frames
        when their distance is <= max_dist. Save plot to PNG.
        NOTE: max_dist is in nm because centroids are stored in nm.
        """
        if len(self.centroid_records) == 0:
            print("No centroid records to plot.")
            return

        from collections import defaultdict

        # organize centroids by frame
        by_frame = defaultdict(list)
        for frame_id, frame_name, nm_per_px, cx_nm, cy_nm in self.centroid_records:
            by_frame[int(frame_id)].append((frame_name, float(cx_nm), float(cy_nm)))

        tracks = []  # each track: {'last_frame': int, 'points': [(frame,cx,cy), ...]}

        for frame in sorted(by_frame.keys()):
            pts = by_frame[frame]
            assigned = [False] * len(pts)

            # try to extend existing tracks from previous frame
            for t in tracks:
                if t['last_frame'] != frame - 1:
                    continue
                last_x, last_y = t['points'][-1][2], t['points'][-1][3]
                best_idx = None
                best_dist = float('inf')
                for i, (frame_name, cx_nm, cy_nm) in enumerate(pts):
                    if assigned[i]:
                        continue
                    d = np.hypot(cx_nm - last_x, cy_nm - last_y)
                    if d < best_dist:
                        best_dist = d
                        best_idx = i

                if best_idx is not None and best_dist <= max_dist:
                    frame_name, cx_nm, cy_nm = pts[best_idx]
                    t['points'].append((frame, frame_name, cx_nm, cy_nm))
                    t['last_frame'] = frame
                    assigned[best_idx] = True

            # create new tracks for unassigned centroids
            for i, (frame_name, cx_nm, cy_nm) in enumerate(pts):
                if not assigned[i]:
                    tracks.append({'last_frame': frame, 'points': [(frame, frame_name, cx_nm, cy_nm)]})

        # plotting
        fig, ax = plt.subplots(figsize=(8, 8))
        scale = float(self.max_nm_per_px) if self.max_nm_per_px is not None else 1.0
        ax.set_xlim(0, self.W * scale * 1.5)
        ax.set_ylim(self.H * scale, 0)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_aspect("equal", adjustable="box")

        # color by frame (time axis) — use same colormap/norm as evolution
        cmap = plt.cm.plasma
        norm = Normalize(vmin=0, vmax=len(self.json_files) - 1)

        for idx, t in enumerate(tracks):
            
            frames = np.array([p[0] for p in t['points']])
            pts = np.array([[p[2], p[3]] for p in t['points']])
            if pts.shape[0] == 0:
                continue

            # draw colored segments between consecutive points according to the earlier frame
            for i in range(len(pts) - 1):
                col = cmap(norm(frames[i]))
                ax.plot(pts[i:i+2, 0], pts[i:i+2, 1], '-', color=col, linewidth=1, alpha=0.95)

            # scatter points colored by their frame
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=frames, cmap=cmap, norm=norm, s=1)

        # add colorbar (time axis)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Frame id colorbar: same height as the axes
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.10)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Frame id")

        ax.set_title(f"{self.gas_category} centroid trajectories (time-colored)", loc="center")
        plt.tight_layout()
        outname = f"{self.gas_category}_centroid_trajectories.png"
        # add a visible border around the axes
        from matplotlib.patches import Rectangle
        border_width = 3
        border_color = "black"
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                 fill=False, edgecolor=border_color,
                 linewidth=border_width, zorder=10, clip_on=False)
        ax.add_patch(rect)
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        print(f"Saved centroid trajectories plot: {outname}")


# ======================
# 主程序入口
# ======================
if __name__ == "__main__":
    tracker = GasTracker(
        json_dir="./data/defect_label",
        image_path="./data/color_mask1121/11dd74426e8374ac110c4036c77c09ab_000000000003.png",
        scale_csv=r"D:\code\nanojccode\data\nanoframes\scalebar_mauel.csv",
        scale_value_nm=20.0,
        strict_scale_match=False,
        gas_category="nanocluster",
        pin_category="pin"
    )
    tracker.process_all_frames()
    tracker.export_results()
    tracker.plot_evolution(step=20)
    tracker.plot_centroid_trajectories(max_dist=50)
    tracker.plot_area_trajectories(max_dist=500, min_track_length=5)
    # 30 fps => 1/30 s per frame; speed unit: nm/s
    tracker.plot_velocity_trajectories(max_dist=50, min_track_length=2, frame_interval_s=1/30, bin_size_frames=1)
