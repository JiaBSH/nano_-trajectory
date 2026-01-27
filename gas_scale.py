import os
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import Normalize


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

    def plot_area_trajectories(self, max_dist=50.0, min_track_length=5, outname=None):
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

        tracks = self._build_greedy_tracks(by_frame, max_dist=max_dist)
        tracks = [t for t in tracks if len(t['points']) >= int(min_track_length)]

        if outname is None:
            outname = f"{self.gas_category}_area_trajectories.png"

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlabel("Frame id")
        ax.set_ylabel("Area (nm^2)")
        ax.grid(True, alpha=0.25)

        # color cycle (good for dozens of tracks; for hundreds, they'll repeat)
        cmap = plt.cm.tab20

        for track_id, t in enumerate(tracks):
            frames = np.array([p[0] for p in t['points']], dtype=np.int32)
            areas = np.array([p[5] for p in t['points']], dtype=np.float32)
            if frames.size == 0:
                continue
            order = np.argsort(frames)
            frames = frames[order]
            areas = areas[order]

            color = cmap(track_id % 20)
            ax.plot(frames, areas, color=color, linewidth=1.2, alpha=0.85)

        plt.title(f"{self.gas_category}: area vs frame (per droplet track)")
        plt.tight_layout()
        plt.savefig(outname, dpi=300)
        print(f"Saved area trajectories plot: {outname}")

        # also export tracked series for downstream analysis
        self.export_tracked_area_results(tracks)
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
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Frame id")

        plt.title(f"{self.gas_category} domain evolution (pin-referenced)")
        plt.tight_layout()
        # add a visible border around the axes
        from matplotlib.patches import Rectangle
        border_width = 3
        border_color = "black"
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                 fill=False, edgecolor=border_color,
                 linewidth=border_width, zorder=10, clip_on=False)
        ax.add_patch(rect)
        plt.savefig(f"{self.gas_category}_evolution.png", dpi=300)

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
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Frame id")

        plt.title(f"{self.gas_category} centroid trajectories (time-colored)")
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
        plt.savefig(outname, dpi=300)
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
    tracker.plot_area_trajectories(max_dist=50, min_track_length=5)
