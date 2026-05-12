import os
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
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
        self.image_dir = os.path.dirname(image_path)
        self.scale_csv = scale_csv
        self.scale_value_nm = float(scale_value_nm)
        self.strict_scale_match = bool(strict_scale_match)
        self.gas_category = gas_category
        self.pin_category = pin_category
        
        # Create output root directory named as gas_category
        self.output_root = self.gas_category
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)

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
        self.diameter_height_records = [] # [frame_id, frame_name, nm_per_px, cx_nm, cy_nm, diameter_nm, height_nm]

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

    @staticmethod
    def _compute_droplet_dims_oriented(pts):
        """
        Compute diameter and height assuming the droplet is a semi-circle projected onto 2D.
        The "base" of the semi-circle is the diameter.
        We find the best-fit bounding box (Minimum Area Rectangle might be too strict if there's noise).
        
        Strategy:
        1. Compute Convex Hull to simplify geometry.
        2. "Bottom" is likely significantly flatter (less curvature) than the "dome".
           OR: The Minimum Area Rectangle's longer side is often the diameter for low contact angles,
           but for high contact angles (>90 deg), the height might be larger than diameter?
           
           Actually, sessile droplets (liquid on solid) generally have a flat contact line.
           This contact line corresponds to one of the sides of the bounding limits.
           
           Let's iterate over all edges of the Convex Hull.
           The edge that is "longest" or creates a bounding box with minimum area is a good candidate.
           However, a fragmented straight line might not be the single longest edge.
           
           Robust Approach:
           - Iterate all edges of the convex hull.
           - Consider the line passing through the edge.
           - Project all points onto this line (width) and perpendicular (height).
           - Metric to maximize: "Linearity" of the points along one side?
           
           Let's stick to the classic "Minimum Area Rectangle" assumption:
           The flat base corresponds to one side of the MAR.
           
           We calculate MAR. It gives us a rectangle with width W and height H and angle theta.
           We need to decide which side is Diameter and which is Height.
           - The "Base" contains the actual contact line.
           - The contact line usually has a high density of points lying very close to the MAR edge.
           
        Returns:
            diameter (float)
            height (float)
            box_info (dict): {
                'box_points': [(x,y)...], # 4 corners of the bounding box
                'baseline_p1': (x,y),
                'baseline_p2': (x,y),
                'apex_point': (x,y)
            }
        """
        if pts.shape[0] < 3:
            return 0.0, 0.0, None

        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
        except Exception:
            return 0.0, 0.0, None
            
        hull_points = pts[hull.vertices]
        num_hull = len(hull_points)
        
        best_metric = float('inf') # We want to minimize volume, or maximize "points on edge"
        best_rect_params = None # (width, height, angle, min_u, min_v, edge_idx)
        
        # Create edges to test
        # Edges of the convex hull are the candidates for the orientation of the bounding box
        edges = hull_points - np.roll(hull_points, 1, axis=0)
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        
        # We will check each edge of the hull as a candidate for the direction of the "flat base".
        results = []
        
        target_angle = np.radians(20) # User specified: ~30 degrees clockwise
        
        for i, angle in enumerate(angles):
            # Rotation matrix to align this edge with X-axis
            c, s = np.cos(-angle), np.sin(-angle)
            R = np.array([[c, -s], [s, c]])
            
            rot_pts = np.dot(pts, R.T)
            
            min_u = np.min(rot_pts[:, 0])
            max_u = np.max(rot_pts[:, 0])
            min_v = np.min(rot_pts[:, 1])
            max_v = np.max(rot_pts[:, 1])
            
            width = max_u - min_u
            height = max_v - min_v
            area = width * height
            
            # Metric: "Closeness to edge".
            # For the correct base, many points (the flat bottom) should be at min_v or max_v.
            # Let's count points within a small epsilon of min_v or max_v.
            tol = height * 0.05 # 5% tolerance
            
            pts_near_bottom = np.sum(rot_pts[:, 1] < (min_v + tol))
            pts_near_top = np.sum(rot_pts[:, 1] > (max_v - tol))
            
            # The base should have MORE points close to it than the apex (which is just a tip).
            base_is_min_v = False
            base_score = 0
            
            if pts_near_bottom > pts_near_top:
                base_score = pts_near_bottom
                base_is_min_v = True
            else:
                base_score = pts_near_top
                base_is_min_v = False
                
            # Angle deviation calculation (modulo pi for line orientation)
            # Shortest angular distance to target_angle
            angle_diff = np.abs(np.arctan2(np.sin(angle - target_angle), np.cos(angle - target_angle)))
            # Since line orientation is symmetric (theta == theta + pi), we want distance to line
            # actually hull edges are vectors. A generic line has range [0, pi).
            # The edge vector angle matches the line angle.
            # But the baseline could be the vector (A->B) or (B->A).
            # If B->A, angle is angle + pi.
            # We want min distance to 30 deg OR (30 + 180) deg.
            angle_diff = min(angle_diff, np.abs(np.pi - angle_diff))
            
            # Penalty factor:
            # If diff is 0, factor = 1.0.
            # If diff is 90 deg (pi/2), factor = MIN_FACTOR (e.g. 0.3)
            # Let's say we heavily penalize perpendicular angles.
            # weighted_score = base_score * (1.0 - 0.7 * (angle_diff / (np.pi/2)))
            weight = 1.0 - 0.7 * (angle_diff / (np.pi / 2))
            weighted_score = base_score * weight
                
            results.append({
                'width': width,
                'height': height,
                'area': area,
                'angle': angle,
                'base_score': base_score,
                'weighted_score': weighted_score,
                'angle_diff': angle_diff,
                'base_is_min_v': base_is_min_v,
                'min_u': min_u, 'max_u': max_u,
                'min_v': min_v, 'max_v': max_v
            })

        # Selection Strategy:
        # Sort by weighted_score descending
        results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # Check if the best score is significantly better than others or if simple Min Area is safer?
        # Let's use a hybrid: Filter for "good" rectangles (low area) then pick max score?
        # Actually, "Max Points on Edge" is very robust for "Flat Line detection".
        best = results[0]
        
        # Reconstruct geometry in original frame
        # We need the baseline endpoints and apex.
        
        angle = best['angle']
        min_u, max_u = best['min_u'], best['max_u']
        min_v, max_v = best['min_v'], best['max_v']
        
        c, s = np.cos(angle), np.sin(angle)
        # Inverse rotation (R^-1 = R^T)
        # R was [[c, -s], [s, c]] (rotation by -angle)
        # We want to map BACK to world.
        # World = Rot * R_inv? 
        # Rot = World * R^T  ==> World = Rot * (R^T)^-1 = Rot * R
        # R_mat for mapping Rot -> World is rotation by +angle
        
        R_back = np.array([[c, -s], [s, c]]) # Wait: c=cos(ang), s=sin(angle). This is rot by angle.
        
        # Corners in rotated space
        # Box is (min_u, min_v) to (max_u, max_v)
        
        # Identify Baseline in rotated space
        if best['base_is_min_v']:
            # Baseline is segment at v = min_v, from u=min_u to max_u
            base_u1, base_v1 = min_u, min_v
            base_u2, base_v2 = max_u, min_v
            
            # Apex is somewhere on v = max_v
            # Let's pick the midpoint of the top edge projected or the actual point with max V?
            # Height is just max_v - min_v
            pass
        else:
            # Baseline is segment at v = max_v
            base_u1, base_v1 = min_u, max_v
            base_u2, base_v2 = max_u, max_v
            pass
            
        # Transform back
        p1 = np.dot(np.array([base_u1, base_v1]), R_back.T) # No, just dot(R_back, vec) or vec.dot(R_back)?
        # rot_pts = np.dot(pts, R.T) -> pts * R_inv
        # R was [[c, -s], [s, c]] (rot by -angle)
        # pts = rot_pts * R_inv^T ? 
        # Let's stick to standard algebra.
        # u = x cos(-a) - y sin(-a) = x c + y s
        # v = x sin(-a) + y cos(-a) = -x s + y c
        #
        # x = u cos(a) - v sin(a)
        # y = u sin(a) + v cos(a)
        # 
        # So trans matrix T = [[ca, -sa], [sa, ca]]
        
        # Actually R above was [[c, -s], [s, c]] for rotation by -angle? 
        # c=cos(-a) = cos(a), s=sin(-a) = -sin(a).
        # So my R construction was:
        # c_val = cos(angle), s_val = sin(angle)
        # c = c_val, s = -s_val
        # R = [[c, -s], [s, c]] = [[cos, sin], [-sin, cos]]
        # This looks like standard rotation matrix for +angle if applied as v' = R v.
        # But I used c, s = cos(-angle), sin(-angle).
        # So theta = -angle.
        # R = [[cos(t), -sin(t)], [sin(t), cos(t)]]
        # v_rot = R . v_world.
        
        # To go back: v_world = R^-1 v_rot = R^T v_rot
        
        # Reconstruct Rotation Matrix used
        ang = -best['angle']
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]])
        
        def to_world(u, v):
            # v_rot = [u, v]
            # v_world = R^T . v_rot
            # v_world = v_rot . R (if row vectors)
            vec = np.array([u, v])
            return np.dot(vec, R)
            
        base_p1_world = to_world(base_u1, base_v1)
        base_p2_world = to_world(base_u2, base_v2)
        
        # Compute diameter and height
        diameter = best['width']
        height = best['height']
        
        # Find apex point in world coordinates
        # Apex is the point maximizing distance from baseline.
        # In rotated frame, if base is at min_v, apex is at max_v.
        # We can find the point with max_v in u-range.
        # Or just return the "Height Line" as per user request (perpendicular from base to top).
        # For visualization, we simply draw the box or the height line.
        # Draw height line from midpoint of base to top?
        
        mid_u = (base_u1 + base_u2) / 2
        mid_v = (base_v1 + base_v2) / 2
        
        if best['base_is_min_v']:
            apex_u, apex_v = mid_u, max_v
        else:
            apex_u, apex_v = mid_u, min_v
            
        apex_world = to_world(apex_u, apex_v)
        base_mid_world = to_world(mid_u, mid_v) # Midpoint on baseline
        
        # Box corners for debug/viz
        c1 = to_world(min_u, min_v)
        c2 = to_world(max_u, min_v)
        c3 = to_world(max_u, max_v)
        c4 = to_world(min_u, max_v)
        
        return diameter, height, {
            'baseline_p1': base_p1_world,
            'baseline_p2': base_p2_world,
            'apex_point': apex_world,
            'base_mid_point': base_mid_world,
            'corners': [c1, c2, c3, c4]
        }

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

            # ---- Diameter and Height (Rotating Calipers / Minimum Area Rectangle) ----
            # The droplet is a semi-circle projected essentially as a "D" shape.
            # The "bottom" is the flat side of the D. 
            # We need to find the orientation of this flat side to measure Diameter (length of flat side)
            # and Height (max perpendicular distance from flat side).
            
            # Use Rotating Calipers via Minimum Area Rectangle to find the major axes.
            # For a semi-circle, the Minimum Area Rectangle usually aligns such that one side is the diameter.
            
            from scipy.spatial import ConvexHull
            


            try:
                # Use the new robust method
                if len(pts) >= 3:
                     # Use the new robust method
                    d_px, h_px, box_info = self._compute_droplet_dims_oriented(pts)
                    d_nm = d_px * nm_per_px
                    h_nm = h_px * nm_per_px
                    
                    self.diameter_height_records.append([
                        frame_id, frame_name, nm_per_px, cx_nm, cy_nm, d_nm, h_nm, box_info
                    ])
                else:
                    self.diameter_height_records.append([frame_id, frame_name, nm_per_px, cx_nm, cy_nm, 0, 0, {}])
            except Exception as e:
                # Fallback to AABB
                print(f"Error in oriented calc: {e}, using AABB")
                min_x, min_y = pts.min(axis=0)
                max_x, max_y = pts.max(axis=0)
                d_nm = (max_x - min_x) * nm_per_px
                h_nm = (max_y - min_y) * nm_per_px
                # Dummy values for the rest
                self.diameter_height_records.append([frame_id, frame_name, nm_per_px, cx_nm, cy_nm, d_nm, h_nm, {}])

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
    def _build_export_instance_ids(self, max_dist=50.0, id_mode="event", use_display_id=True):
        """Build per-record droplet ids aligned with object_records order."""
        if len(self.object_records) == 0:
            return []

        from collections import defaultdict

        by_frame = defaultdict(list)
        for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in self.object_records:
            by_frame[int(frame_id)].append((frame_name, float(nm_per_px), float(cx_nm), float(cy_nm), float(area_nm2)))

        mode = str(id_mode).strip().lower()
        if mode != "event":
            raise NotImplementedError("export_results currently supports id_mode='event' only")

        series_by_id, assigned_ids_by_frame, _events = self._build_event_id_series_with_assignments(
            by_frame,
            max_dist=max_dist,
        )

        if bool(use_display_id):
            display_id_of = self._display_id_mapping(series_by_id)
            assigned_ids_by_frame = {
                int(frame_id): [int(display_id_of.get(int(instance_id), int(instance_id))) for instance_id in ids]
                for frame_id, ids in assigned_ids_by_frame.items()
            }

        export_ids = []
        for frame_id in sorted(assigned_ids_by_frame.keys()):
            export_ids.extend(assigned_ids_by_frame[frame_id])

        if len(export_ids) != len(self.object_records):
            raise ValueError(
                f"Export instance-id count mismatch: ids={len(export_ids)} object_records={len(self.object_records)}"
            )

        return export_ids

    def export_results(self, max_dist=50.0, id_mode="event", use_display_id=True):
        export_ids = self._build_export_instance_ids(
            max_dist=max_dist,
            id_mode=id_mode,
            use_display_id=use_display_id,
        )

        # 面积
        path1 = os.path.join(self.output_root, f"{self.gas_category}_area_vs_frame.csv")
        with open(path1, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "frame_id", "frame_name", "nm_per_pixel", "area_nm2"])
            writer.writerows(
                [[int(instance_id), frame_id, frame_name, f"{nm_per_px:.6f}", f"{area_nm2:.6f}"]
                 for instance_id, (frame_id, frame_name, nm_per_px, area_nm2) in zip(export_ids, self.area_records)]
            )

        # 轮廓（每帧一行）
        path2 = os.path.join(self.output_root, f"{self.gas_category}_contours_by_frame.csv")
        with open(path2, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "frame_id", "frame_name", "contour_points_nm"])
            writer.writerows(
                [[int(instance_id)] + row for instance_id, row in zip(export_ids, self.contour_records)]
            )

        # 质心
        path3 = os.path.join(self.output_root, f"{self.gas_category}_centroids.csv")
        with open(path3, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "frame_id", "frame_name", "nm_per_pixel", "cx_nm", "cy_nm"])
            writer.writerows(
                [[int(instance_id), frame_id, frame_name, f"{nm_per_px:.6f}", f"{cx_nm:.6f}", f"{cy_nm:.6f}"]
                 for instance_id, (frame_id, frame_name, nm_per_px, cx_nm, cy_nm) in zip(export_ids, self.centroid_records)]
            )

        # Diameter and Height
        path4 = os.path.join(self.output_root, f"{self.gas_category}_diameter_height_vs_frame.csv")
        with open(path4, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "frame_id", "frame_name", "nm_per_pixel", "cx_nm", "cy_nm", "diameter_nm", "height_nm"])
            for instance_id, row in zip(export_ids, self.diameter_height_records):
                # row structure: [frame_id, frame_name, nm_per_px, cx_nm, cy_nm, d_nm, h_nm, min_x, min_y, max_x, max_y]
                # we only export the first 7 fields here
                writer.writerow([int(instance_id), row[0], row[1], f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.6f}", f"{row[5]:.6f}", f"{row[6]:.6f}"])

        print("Export finished:")
        print(f" - {path1}")
        print(f" - {path2}")
        print(f" - {path3}")
        print(f" - {path4}")

    def annotate_images(
        self,
        output_dir=None,
        label_ids=False,
        id_mode="event",
        max_dist=50.0,
        min_track_length=0,
        use_display_id=True,
    ):
        if output_dir is None:
             output_dir = os.path.join(self.output_root, "annotated_images")
        else:
             # If user supplied output_dir is absolute, use it. Else join with output_root?
             # Let's assume user supplied just a name like "annotated_nanodroplet" and we want it inside output_root
             # Check if it looks like an absolute path
             if not os.path.isabs(output_dir):
                 output_dir = os.path.join(self.output_root, output_dir)
                 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Annotating images to {output_dir}...")
        
        try:
            # Try to start with a slightly larger font if possible
            font = ImageFont.truetype("arial.ttf", 24)
        except OSError:
            font = ImageFont.load_default()

        assigned_ids_by_frame = None
        display_id_of = None
        if bool(label_ids):
            if len(self.object_records) == 0:
                print("[warn] label_ids=True but object_records is empty; run process_all_frames() first.")
            else:
                from collections import defaultdict

                detections_by_frame = defaultdict(list)
                for frame_id, frame_name, nm_per_px, cx_nm, cy_nm, area_nm2 in self.object_records:
                    detections_by_frame[int(frame_id)].append(
                        (str(frame_name), float(nm_per_px), float(cx_nm), float(cy_nm), float(area_nm2))
                    )

                mode = str(id_mode).strip().lower()
                if mode != "event":
                    raise NotImplementedError("annotate_images(label_ids=True) currently supports id_mode='event' only")

                series_by_id, assigned_ids_by_frame, _events = self._build_event_id_series_with_assignments(
                    detections_by_frame, max_dist=max_dist
                )

                series_by_id_for_display = {
                    k: v for k, v in series_by_id.items() if len(v) >= int(min_track_length)
                }
                if bool(use_display_id):
                    display_id_of = self._display_id_mapping(series_by_id_for_display)
                else:
                    display_id_of = None

                print(
                    f"Annotate IDs enabled: mode={mode}, max_dist_nm={float(max_dist)}, "
                    f"min_track_length={int(min_track_length)}, use_display_id={bool(use_display_id)}, "
                    f"ids_total={len(series_by_id)}"
                )

        # For robust annotation across categories:
        # - Draw the segmentation contours for self.gas_category.
        # - Only for nanodroplet, additionally draw diameter/height overlays.
        for frame_id, json_name in enumerate(self.json_files):
            frame_name = Path(json_name).stem
            # Find image
            img_path = None
            possible_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
            for ext in possible_exts:
                p = os.path.join(self.image_dir, frame_name + ext)
                if os.path.exists(p):
                    img_path = p
                    break
            if not img_path:
                continue

            try:
                with Image.open(img_path) as im:
                    img_out = im.convert("RGB")
                    draw = ImageDraw.Draw(img_out)

                    json_path = os.path.join(self.json_dir, json_name)
                    with open(json_path, "r", encoding="utf-8") as f:
                        jdata = json.load(f)

                    try:
                        nm_per_px = self._nm_per_px_for_frame(frame_name)
                    except Exception:
                        nm_per_px = None

                    ids_this_frame = None
                    if assigned_ids_by_frame is not None:
                        ids_this_frame = assigned_ids_by_frame.get(int(frame_id))

                    obj_idx = 0

                    for obj in jdata.get("objects", []):
                        if obj.get("category") != self.gas_category:
                            continue
                        pts_raw = np.array(obj.get("segmentation", []), dtype=np.float32)
                        if pts_raw.shape[0] < 3:
                            obj_idx += 1
                            continue

                        # ID label (use the same within-frame order as JSON/category iteration)
                        if bool(label_ids) and ids_this_frame is not None and obj_idx < len(ids_this_frame):
                            try:
                                instance_id = int(ids_this_frame[obj_idx])
                                if display_id_of is not None:
                                    disp = int(display_id_of.get(instance_id, 0))
                                    id_text = str(disp) if disp > 0 else str(instance_id)
                                else:
                                    id_text = str(instance_id)

                                cx_px = float(np.mean(pts_raw[:, 0]))
                                cy_px = float(np.mean(pts_raw[:, 1]))
                                r = 6
                                draw.ellipse((cx_px - r, cy_px - r, cx_px + r, cy_px + r), outline="orange", width=3)
                                draw.text(
                                    (cx_px - 20, cy_px -18),
                                    id_text,
                                    fill="orange",
                                    font=font,
                                    stroke_width=2,
                                    stroke_fill="black",
                                )
                            except Exception:
                                pass

                        # draw contour
                        poly = [tuple(map(float, p)) for p in pts_raw]
                        draw.polygon(poly, outline="lime", width=2)

                        # droplet-only: draw diameter/height overlay
                        if str(self.gas_category).lower() == "nanodroplet" and nm_per_px is not None:
                            try:
                                d_px, h_px, box_info = self._compute_droplet_dims_oriented(pts_raw)
                                d_nm = float(d_px) * float(nm_per_px)
                                h_nm = float(h_px) * float(nm_per_px)

                                corners = box_info.get("corners")
                                if corners is not None:
                                    corners = np.array(corners, dtype=np.float32)
                                    rect_poly = [tuple(map(float, p)) for p in corners]
                                    draw.polygon(rect_poly, outline="cyan", width=2)

                                    text = f"D:{d_nm:.1f}\nH:{h_nm:.1f}"
                                    cx = float(corners[:, 0].mean())
                                    cy = float(corners[:, 1].mean())
                                    draw.text((cx, cy), text, fill="yellow", font=font)

                                baseline_p1 = box_info.get("baseline_p1")
                                baseline_p2 = box_info.get("baseline_p2")
                                if baseline_p1 is not None and baseline_p2 is not None:
                                    draw.line([tuple(map(float, baseline_p1)), tuple(map(float, baseline_p2))], fill="red", width=3)

                                apex = box_info.get("apex_point")
                                base_mid = box_info.get("base_mid_point")
                                if apex is not None and base_mid is not None:
                                    draw.line([tuple(map(float, apex)), tuple(map(float, base_mid))], fill="magenta", width=2)
                            except Exception:
                                # best-effort; keep contour even if dims fail
                                pass

                        obj_idx += 1

                    out_path = os.path.join(output_dir, frame_name + ".png")
                    img_out.save(out_path)

            except Exception as e:
                print(f"Error annotating {frame_name}: {e}")

    def export_tracked_area_results(self, tracks, out_csv=None):
        """Export tracked area series.

        CSV columns: track_id, frame_id, frame_name, nm_per_pixel, area_nm2, cx_nm, cy_nm
        """
        if out_csv is None:
            out_csv = os.path.join(self.output_root, f"{self.gas_category}_tracked_area_vs_frame.csv")
        elif not os.path.isabs(out_csv):
             out_csv = os.path.join(self.output_root, out_csv)

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
            out_csv = os.path.join(self.output_root, f"{self.gas_category}_instance_area_vs_frame.csv")
        elif not os.path.isabs(out_csv):
             out_csv = os.path.join(self.output_root, out_csv)

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
            out_csv = os.path.join(self.output_root, f"{self.gas_category}_instance_speed_vs_frame.csv")
        elif not os.path.isabs(out_csv):
             out_csv = os.path.join(self.output_root, out_csv)

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

    def _build_event_id_series(self, detections_by_frame, max_dist=50.0, return_assignments=False):
        """Assign globally-incrementing ids with merge/split relabeling.

        Rules:
        - First frame detections get ids 1..N
        - Merge (many prev -> one curr): curr gets a NEW id
        - Split (one prev -> many curr): each child gets a NEW id
        - 1-to-1 continuation keeps the same id

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

        if bool(return_assignments):
            return dict(series_by_id), assigned_ids_by_frame, events
        return dict(series_by_id), events

    def _build_event_id_series_with_assignments(self, detections_by_frame, max_dist=50.0):
        """Compatibility helper: return series + per-frame assignment list + events."""
        return self._build_event_id_series(detections_by_frame, max_dist=max_dist, return_assignments=True)

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

    def plot_area_trajectories(self, max_dist=50.0, min_track_length=1, outname=None, id_mode="event", debug_stats=False):
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

        if bool(debug_stats):
            n_frames = len(by_frame)
            n_dets = sum(len(v) for v in by_frame.values())
            print(
                f"[debug] {self.gas_category} area: frames_with_detections={n_frames}, total_detections={n_dets}, "
                f"id_mode={id_mode}, max_dist_nm={float(max_dist)}, min_track_length={int(min_track_length)}"
            )

        if str(id_mode).lower() == "greedy":
            tracks = self._build_greedy_tracks(by_frame, max_dist=max_dist)
            if bool(debug_stats):
                print(f"[debug] {self.gas_category} area: greedy tracks before length filter={len(tracks)}")
            tracks = [t for t in tracks if len(t['points']) >= int(min_track_length)]
            if bool(debug_stats):
                print(f"[debug] {self.gas_category} area: greedy tracks after length filter={len(tracks)}")
            series_by_id = {int(track_id): [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in t["points"]] for track_id, t in enumerate(tracks)}
        else:
            series_by_id, _events = self._build_event_id_series(by_frame, max_dist=max_dist)
            if bool(debug_stats):
                print(f"[debug] {self.gas_category} area: event ids before length filter={len(series_by_id)}")
            series_by_id = {k: v for k, v in series_by_id.items() if len(v) >= int(min_track_length)}
            if bool(debug_stats):
                kept = len(series_by_id)
                max_iid = max(series_by_id.keys()) if kept > 0 else None
                print(f"[debug] {self.gas_category} area: event ids after length filter={kept}, max_instance_id={max_iid}")

        if outname is None:
            outname = os.path.join(self.output_root, f"{self.gas_category}_area_trajectories.png")
        elif not os.path.isabs(outname):
            outname = os.path.join(self.output_root, outname)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlabel("Frame id")
        ax.set_ylabel("Area (nm^2)")
        ax.grid(True, alpha=0.25)

        # color cycle (good for dozens of tracks; for hundreds, they'll repeat)
        cmap = plt.cm.tab20

        display_id_of = self._display_id_mapping(series_by_id)
        instance_ids = sorted(series_by_id.keys())

        if bool(debug_stats):
            max_disp = max(display_id_of.values()) if len(display_id_of) > 0 else 0
            print(f"[debug] {self.gas_category} area: plotted_ids={len(instance_ids)}, display_id_max={max_disp}")

        line_handles = []
        line_labels = []

        skipped_empty = []

        for instance_id in instance_ids:
            pts = series_by_id[instance_id]
            frames = np.array([p[0] for p in pts], dtype=np.int32)
            areas = np.array([p[5] for p in pts], dtype=np.float32)
            if frames.size == 0:
                if bool(debug_stats):
                    skipped_empty.append(int(instance_id))
                continue
            order = np.argsort(frames)
            frames = frames[order]
            areas = areas[order]

            disp_id = display_id_of.get(int(instance_id), 0)
            color = cmap(int(disp_id) % 20)
            (line,) = ax.plot(frames, areas, color=color, linewidth=1.2, alpha=0.85)
            line_handles.append(line)
            line_id_label = str(int(disp_id) if disp_id > 0 else int(instance_id))
            line_labels.append(line_id_label)

            # Mark ID at the start of each curve
            try:
                x0 = float(frames[0])
                y0 = float(areas[0])
                ax.annotate(
                    line_id_label,
                    xy=(x0, y0),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    color=color,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
                )
            except Exception:
                pass

        if bool(debug_stats):
            if len(skipped_empty) > 0:
                print(f"[debug] {self.gas_category} area: skipped_empty_ids={len(skipped_empty)} head={skipped_empty[:20]}")
            print(f"[debug] {self.gas_category} area: drawn_lines={len(line_handles)} legend_items={len(line_labels)}")

        # 图例：自适应布局，避免挡线 & 避免图被挤得很“扁”
        leg = None
        if len(line_handles) > 0:
            n_items = len(line_handles)

            # Prefer right-side legend for moderate counts; switch to multi-column / bottom for very long legends.
            if n_items <= 20:
                ncol = 1
                fig.set_size_inches(12, 6, forward=True)
                fig.subplots_adjust(right=0.80)
                leg = ax.legend(
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
                leg = ax.legend(
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
                # Too many: put legend below with more columns to avoid clipping.
                # Aim for <= ~12 rows in legend.
                rows_target = 12
                ncol = int(np.ceil(float(n_items) / float(rows_target)))
                ncol = max(4, min(10, ncol))
                fig.set_size_inches(16, 11.0, forward=True)
                fig.subplots_adjust(bottom=0.36)
                leg = ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.22),
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=6,
                    ncol=ncol,
                    columnspacing=0.8,
                    handlelength=1.2,
                )

        ax.set_title(
            f"{self.gas_category}: area vs frame (per droplet track) | tracks={len(instance_ids)}",
            loc="center",
        )
        plt.tight_layout()
        plt.savefig(outname, dpi=300, bbox_inches="tight", bbox_extra_artists=((leg,) if leg is not None else None))
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
        debug_stats=False,
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

        if bool(debug_stats):
            n_frames = len(by_frame)
            n_dets = sum(len(v) for v in by_frame.values())
            print(
                f"[debug] {self.gas_category} speed: frames_with_detections={n_frames}, total_detections={n_dets}, "
                f"id_mode={id_mode}, max_dist_nm={float(max_dist)}, min_track_length={int(min_track_length)}, "
                f"frame_interval_s={float(frame_interval_s)}, bin_size_frames={int(bin_size_frames)}"
            )

        if str(id_mode).lower() == "greedy":
            tracks = self._build_greedy_tracks(by_frame, max_dist=max_dist)
            if bool(debug_stats):
                print(f"[debug] {self.gas_category} speed: greedy tracks before length filter={len(tracks)}")
            tracks = [t for t in tracks if len(t["points"]) >= int(min_track_length)]
            if bool(debug_stats):
                print(f"[debug] {self.gas_category} speed: greedy tracks after length filter={len(tracks)}")
            series_by_id = {
                int(track_id): [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in t["points"]]
                for track_id, t in enumerate(tracks)
            }
        else:
            series_by_id, _events = self._build_event_id_series(by_frame, max_dist=max_dist)
            if bool(debug_stats):
                print(f"[debug] {self.gas_category} speed: event ids before length filter={len(series_by_id)}")
            series_by_id = {k: v for k, v in series_by_id.items() if len(v) >= int(min_track_length)}
            if bool(debug_stats):
                print(f"[debug] {self.gas_category} speed: event ids after length filter={len(series_by_id)}")

        # compute speed series for each id (raw, per-frame)
        speed_series_by_id = {}
        empty_speed_ids = 0
        for instance_id, pts in series_by_id.items():
            sp = self._compute_speed_series_from_points(pts, frame_interval_s=frame_interval_s)
            if len(sp) > 0:
                speed_series_by_id[int(instance_id)] = sp
            else:
                empty_speed_ids += 1

        if bool(debug_stats):
            print(
                f"[debug] {self.gas_category} speed: ids_with_speed={len(speed_series_by_id)}, "
                f"ids_dropped_empty_speed={int(empty_speed_ids)} (typically tracks with <2 detections)"
            )

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

        if bool(debug_stats):
            max_disp = 0
            if len(binned_speed_by_id) > 0:
                display_id_of_dbg = self._display_id_mapping(binned_speed_by_id)
                max_disp = max(display_id_of_dbg.values()) if len(display_id_of_dbg) > 0 else 0
            print(f"[debug] {self.gas_category} speed: plotted_ids={len(binned_speed_by_id)}, display_id_max={max_disp}")

        if outname is None:
            if b == 1:
                outname = os.path.join(self.output_root, f"{self.gas_category}_velocity_trajectories.png")
            else:
                outname = os.path.join(self.output_root, f"{self.gas_category}_velocity_mean_{b}frames.png")
        elif not os.path.isabs(outname):
            outname = os.path.join(self.output_root, outname)

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
            line_id_label = str(int(disp_id) if disp_id > 0 else int(instance_id))
            line_labels.append(line_id_label)

            # Mark ID at the start of each curve
            try:
                x0 = float(frames[0])
                y0 = float(speeds[0])
                ax.annotate(
                    line_id_label,
                    xy=(x0, y0),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    color=color,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
                )
            except Exception:
                pass

        # legend layout (same idea as area plot)
        leg = None
        if len(line_handles) > 0:
            n_items = len(line_handles)
            if n_items <= 20:
                fig.set_size_inches(12, 6, forward=True)
                fig.subplots_adjust(right=0.80)
                leg = ax.legend(
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
                leg = ax.legend(
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
                rows_target = 12
                ncol = int(np.ceil(float(n_items) / float(rows_target)))
                ncol = max(4, min(10, ncol))
                fig.set_size_inches(16, 11.0, forward=True)
                fig.subplots_adjust(bottom=0.36)
                leg = ax.legend(
                    handles=line_handles,
                    labels=line_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.22),
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="gray",
                    fontsize=6,
                    ncol=ncol,
                    columnspacing=0.8,
                    handlelength=1.2,
                )

        if b == 1:
            ax.set_title(f"{self.gas_category}: velocity vs frame (per track) | tracks={len(instance_ids)}", loc="center")
        else:
            ax.set_title(f"{self.gas_category}: mean velocity per {b} frames | tracks={len(instance_ids)}", loc="center")
        plt.tight_layout()
        plt.savefig(outname, dpi=300, bbox_inches="tight", bbox_extra_artists=((leg,) if leg is not None else None))
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

    def plot_area_delta_vs_frame(
        self,
        outname=None,
        out_csv=None,
        per_frame=True,
        reducer="sum",
    ):
        """Plot per-frame change in area (Δarea) as a single curve.

        This is computed from `self.area_records` by first aggregating all detections within
        the same frame (default: sum), then taking a first-order difference between
        consecutive frames.

        Args:
            outname: Output PNG name.
            out_csv: Optional CSV output for the delta series.
            per_frame: If True, normalize by delta_frame (handles skipped frames).
            reducer: How to aggregate multiple objects in the same frame: 'sum' or 'mean'.
        """
        if len(self.area_records) == 0:
            print("No area records to plot area delta.")
            return

        from collections import defaultdict

        reducer_key = str(reducer).strip().lower()
        if reducer_key not in {"sum", "mean"}:
            raise ValueError(f"reducer must be 'sum' or 'mean', got {reducer}")

        areas_by_frame = defaultdict(list)  # frame_id -> list[area_nm2]
        name_by_frame = {}
        for frame_id, frame_name, _nm_per_px, area_nm2 in self.area_records:
            fid = int(frame_id)
            areas_by_frame[fid].append(float(area_nm2))
            if fid not in name_by_frame:
                name_by_frame[fid] = str(frame_name)

        frame_ids = sorted(areas_by_frame.keys())
        if len(frame_ids) < 2:
            print("Not enough frames to compute area delta (need >= 2).")
            return

        area_series = []  # (frame_id, frame_name, area_agg_nm2)
        for fid in frame_ids:
            vals = areas_by_frame[fid]
            if len(vals) == 0:
                continue
            if reducer_key == "mean":
                a = float(np.mean(vals))
            else:
                a = float(np.sum(vals))
            area_series.append((int(fid), name_by_frame.get(int(fid), str(fid)), a))

        # ensure sorted
        area_series.sort(key=lambda t: int(t[0]))

        # delta aligned to current frame
        delta_points = []  # (frame_id, frame_name, delta_area_nm2_per_frame)
        prev_f, _prev_name, prev_a = area_series[0]
        for cur_f, cur_name, cur_a in area_series[1:]:
            df = int(cur_f) - int(prev_f)
            if df <= 0:
                prev_f, prev_a = cur_f, cur_a
                continue
            da = float(cur_a) - float(prev_a)
            if bool(per_frame):
                da = da / float(df)
            delta_points.append((int(cur_f), str(cur_name), float(da)))
            prev_f, prev_a = cur_f, cur_a

        if len(delta_points) == 0:
            print("Area delta series is empty after processing.")
            return

        if outname is None:
            outname = os.path.join(self.output_root, f"{self.gas_category}_area_delta_vs_frame.png")
        elif not os.path.isabs(outname):
            outname = os.path.join(self.output_root, outname)

        if out_csv is None:
            out_csv = os.path.join(self.output_root, f"{self.gas_category}_area_delta_vs_frame.csv")
        elif not os.path.isabs(out_csv):
            out_csv = os.path.join(self.output_root, out_csv)

        frames = np.array([p[0] for p in delta_points], dtype=np.int32)
        deltas = np.array([p[2] for p in delta_points], dtype=np.float64)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(frames, deltas, color="#1f77b4", linewidth=1.6)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.set_xlabel("Frame id")
        ylab = "ΔArea (nm^2/frame)" if bool(per_frame) else "ΔArea (nm^2)"
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)

        agg_label = "sum" if reducer_key == "sum" else "mean"
        ax.set_title(f"{self.gas_category}: per-frame area change (Δarea), frame-agg={agg_label}", loc="center")

        plt.tight_layout()
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        print(f"Saved area delta plot: {outname}")

        # export delta CSV
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "frame_name", "delta_area_nm2_per_frame" if bool(per_frame) else "delta_area_nm2"])
            for fid, fname, da in delta_points:
                writer.writerow([int(fid), str(fname), f"{float(da):.6f}"])
        print(f" - {out_csv}")
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
        outname = os.path.join(self.output_root, f"{self.gas_category}_evolution.png")
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        print(f"Saved evolution plot: {outname}")

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
        outname = os.path.join(self.output_root, f"{self.gas_category}_centroid_trajectories.png")
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
        json_dir="./data/20260508-mark",
        image_path="./data/20260508-mark-color/11dd74426e8374ac110c4036c77c09ab_000000000003.png",
        scale_csv=r"D:\code\nanojccode\data\nanoframes\scalebar_mauel.csv",
        #output_root="./result/0510",
        scale_value_nm=20.0,
        strict_scale_match=False,
        gas_category="gas",
        pin_category="pin"
    )
    tracker.process_all_frames()
    tracker.export_results()
    # Output dir logic now inside class if passed None, or relative to output_root if passed string
    tracker.annotate_images(output_dir="annotated_gas",label_ids=True) 
    tracker.plot_evolution(step=20)
    tracker.plot_centroid_trajectories(max_dist=50)
    tracker.plot_area_trajectories(max_dist=50, min_track_length=0, debug_stats=True)
    tracker.plot_area_delta_vs_frame(per_frame=True, reducer="sum")
    # 30 fps => 1/30 s per frame; speed unit: nm/s
    tracker.plot_velocity_trajectories(max_dist=50, min_track_length=0, frame_interval_s=1/30, bin_size_frames=1, debug_stats=True)
