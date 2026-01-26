import os
import json
import csv
from matplotlib import category
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import Normalize


class GasTracker:
    def __init__(
        self,
        json_dir,
        image_path,
        gas_category="gas",
        pin_category="pin"
    ):
        self.json_dir = json_dir
        self.image_path = image_path
        self.gas_category = gas_category
        self.pin_category = pin_category

        self.json_files = self._load_and_sort_jsons()

        # 数据容器
        self.area_records = []        # [frame, area]
        self.contour_records = []     # [frame, "(x1,y1)", "(x2,y2)", ...]
        self.centroid_records = []    # [frame, cx, cy]
        self.object_records = []      # [frame, cx, cy, area]

        # pin 参考
        self.ref_pin_centroid = None
        self.last_shift = np.zeros(2)

        # 画图准备
        img = Image.open(image_path)
        self.W, self.H = img.size

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

            shift = self._compute_pin_shift(data)

            self._process_gas_objects(
                data,
                frame_id,
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

    def _process_gas_objects(self, data, frame_id, shift):
        for obj in data.get("objects", []):
            if obj.get("category") != self.gas_category:
                continue

            pts = np.array(obj["segmentation"], dtype=np.float32)
            pts = pts - shift   # ★ 去整体漂移

            if pts.shape[0] < 3:
                continue

            # ---- 面积 ----
            area = self.polygon_area(pts)
            self.area_records.append([frame_id, area])

            # ---- 质心 ----
            centroid = pts.mean(axis=0)
            cx, cy = float(centroid[0]), float(centroid[1])
            self.centroid_records.append([frame_id, cx, cy])

            # ---- 每个目标的聚合记录（用于追踪面积曲线）----
            self.object_records.append([frame_id, cx, cy, area])

            # ---- 轮廓（每帧一行）----
            row = [frame_id]
            for (x, y) in pts:
                row.append(f"({x:.2f},{y:.2f})")
            self.contour_records.append(row)

    # -----------------------------
    # 数据导出
    # -----------------------------
    def export_results(self):
        # 面积
        with open(f"{self.gas_category}_area_vs_frame.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "area"])
            writer.writerows([[frame, f"{area:.2f}"] for frame, area in self.area_records])

        # 轮廓（每帧一行）
        with open(f"{self.gas_category}_contours_by_frame.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(self.contour_records)

        # 质心
        with open(f"{self.gas_category}_centroids.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "cx", "cy"])
            writer.writerows([[frame, f"{cx:.2f}", f"{cy:.2f}"] for frame, cx, cy in self.centroid_records])

        print("Export finished:")
        print(f" - {self.gas_category}_area_vs_frame.csv")
        print(f" - {self.gas_category}_contours_by_frame.csv")

    def export_tracked_area_results(self, tracks, out_csv=None):
        """Export tracked area series.

        CSV columns: track_id, frame, area, cx, cy
        """
        if out_csv is None:
            out_csv = f"{self.gas_category}_tracked_area_vs_frame.csv"

        rows = []
        for track_id, t in enumerate(tracks):
            for frame, cx, cy, area in t['points']:
                rows.append([track_id, frame, f"{area:.2f}", f"{cx:.2f}", f"{cy:.2f}"])

        rows.sort(key=lambda r: (r[0], r[1]))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "frame", "area", "cx", "cy"])
            writer.writerows(rows)

        print(f" - {out_csv}")

    def _build_greedy_tracks(self, detections_by_frame, max_dist=50):
        """Greedy link detections in consecutive frames into tracks.

        detections_by_frame: dict[int, list[tuple[cx,cy,area]]]
        returns: list of tracks, each track: {'last_frame': int, 'points': [(frame,cx,cy,area), ...]}
        """
        tracks = []
        for frame in sorted(detections_by_frame.keys()):
            dets = detections_by_frame[frame]
            assigned = [False] * len(dets)

            # extend tracks from previous frame
            for t in tracks:
                if t['last_frame'] != frame - 1:
                    continue

                last_x, last_y = t['points'][-1][1], t['points'][-1][2]
                best_idx = None
                best_dist = float('inf')
                for i, (cx, cy, area) in enumerate(dets):
                    if assigned[i]:
                        continue
                    d = np.hypot(cx - last_x, cy - last_y)
                    if d < best_dist:
                        best_dist = d
                        best_idx = i

                if best_idx is not None and best_dist <= max_dist:
                    cx, cy, area = dets[best_idx]
                    t['points'].append((frame, cx, cy, area))
                    t['last_frame'] = frame
                    assigned[best_idx] = True

            # create new tracks for unassigned detections
            for i, (cx, cy, area) in enumerate(dets):
                if not assigned[i]:
                    tracks.append({'last_frame': frame, 'points': [(frame, cx, cy, area)]})

        return tracks

    def plot_area_trajectories(self, max_dist=50, min_track_length=5, outname=None):
        """Plot each droplet's area-vs-frame curve in one figure.

        Tracks are built by greedy centroid linking (same logic as centroid trajectories).
        """
        if len(self.object_records) == 0:
            print("No object records to plot area trajectories.")
            return

        from collections import defaultdict

        by_frame = defaultdict(list)
        for frame, cx, cy, area in self.object_records:
            by_frame[int(frame)].append((float(cx), float(cy), float(area)))

        tracks = self._build_greedy_tracks(by_frame, max_dist=max_dist)
        tracks = [t for t in tracks if len(t['points']) >= int(min_track_length)]

        if outname is None:
            outname = f"{self.gas_category}_area_trajectories.png"

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlabel("Frame")
        ax.set_ylabel("Area (px^2)")
        ax.grid(True, alpha=0.25)

        # color cycle (good for dozens of tracks; for hundreds, they'll repeat)
        cmap = plt.cm.tab20

        for track_id, t in enumerate(tracks):
            frames = np.array([p[0] for p in t['points']], dtype=np.int32)
            areas = np.array([p[3] for p in t['points']], dtype=np.float32)
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
        ax.set_xlim(0, self.W * 1.5)
        ax.set_ylim(self.H, 0)
        ax.axis("off")

        cmap = plt.cm.plasma
        norm = Normalize(vmin=0, vmax=len(self.json_files) - 1)

        for row in self.contour_records:
            frame_id = row[0]
            if frame_id % step != 0:
                continue

            pts = []
            for item in row[1:]:
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
        cbar.set_label("Frame index")

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

    def plot_centroid_trajectories(self, max_dist=50):
        """
        Build simple greedy tracks by linking centroids in consecutive frames
        when their distance is <= max_dist. Save plot to PNG.
        """
        if len(self.centroid_records) == 0:
            print("No centroid records to plot.")
            return

        from collections import defaultdict

        # organize centroids by frame
        by_frame = defaultdict(list)
        for frame, cx, cy in self.centroid_records:
            by_frame[frame].append((cx, cy))

        tracks = []  # each track: {'last_frame': int, 'points': [(frame,cx,cy), ...]}

        for frame in sorted(by_frame.keys()):
            pts = by_frame[frame]
            assigned = [False] * len(pts)

            # try to extend existing tracks from previous frame
            for t in tracks:
                if t['last_frame'] != frame - 1:
                    continue
                last_x, last_y = t['points'][-1][1], t['points'][-1][2]
                best_idx = None
                best_dist = float('inf')
                for i, (cx, cy) in enumerate(pts):
                    if assigned[i]:
                        continue
                    d = np.hypot(cx - last_x, cy - last_y)
                    if d < best_dist:
                        best_dist = d
                        best_idx = i

                if best_idx is not None and best_dist <= max_dist:
                    cx, cy = pts[best_idx]
                    t['points'].append((frame, cx, cy))
                    t['last_frame'] = frame
                    assigned[best_idx] = True

            # create new tracks for unassigned centroids
            for i, (cx, cy) in enumerate(pts):
                if not assigned[i]:
                    tracks.append({'last_frame': frame, 'points': [(frame, cx, cy)]})

        # plotting
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.W * 1.5)
        ax.set_ylim(self.H, 0)
        ax.axis("off")

        # color by frame (time axis) — use same colormap/norm as evolution
        cmap = plt.cm.plasma
        norm = Normalize(vmin=0, vmax=len(self.json_files) - 1)

        for idx, t in enumerate(tracks):
            
            frames = np.array([p[0] for p in t['points']])
            pts = np.array([[p[1], p[2]] for p in t['points']])
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
        cbar.set_label("Frame index")

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
        gas_category="nanodroplet",
        pin_category="pin"
    )
    tracker.process_all_frames()
    tracker.export_results()
    tracker.plot_evolution(step=50)
    tracker.plot_centroid_trajectories(max_dist=50)
    tracker.plot_area_trajectories(max_dist=50, min_track_length=5)
