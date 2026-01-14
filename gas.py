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
                return

            # ---- 面积 ----
            area = self.polygon_area(pts)
            self.area_records.append([frame_id, area])

            # ---- 轮廓（每帧一行）----
            row = [frame_id]
            for (x, y) in pts:
                row.append(f"({x:.3f},{y:.3f})")
            self.contour_records.append(row)

    # -----------------------------
    # 数据导出
    # -----------------------------
    def export_results(self):
        # 面积
        with open(f"{self.gas_category}_area_vs_frame.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "area"])
            writer.writerows(self.area_records)

        # 轮廓（每帧一行）
        with open(f"{self.gas_category}_contours_by_frame.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(self.contour_records)

        print("Export finished:")
        print(f" - {self.gas_category}_area_vs_frame.csv")
        print(f" - {self.gas_category}_contours_by_frame.csv")
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
        plt.savefig(f"{self.gas_category}_evolution.png", dpi=300)


# ======================
# 主程序入口
# ======================
if __name__ == "__main__":
    tracker = GasTracker(
        json_dir="./defect_label",
        image_path="./color_mask1121/11dd74426e8374ac110c4036c77c09ab_000000000003.png",
        gas_category="nanocluster",
        pin_category="pin"
    )
    tracker.process_all_frames()
    tracker.export_results()
    tracker.plot_evolution(step=20)
