"""Polygon and droplet geometry calculations."""

from __future__ import annotations

import numpy as np


class GeometryMixin:
    """Provide polygon and droplet geometry calculations."""

    @staticmethod
    def polygon_boundary_distance(polygon_a, polygon_b):
        """Return the shortest distance between two filled polygons.

        The result uses the same coordinate unit as the inputs.  Intersecting,
        touching, or nested polygons have distance zero; otherwise the result is
        the minimum Euclidean distance between their closed boundaries.
        """
        a = GeometryMixin._distance_polygon_points(polygon_a, "polygon_a")
        b = GeometryMixin._distance_polygon_points(polygon_b, "polygon_b")
        coordinate_scale = max(
            1.0,
            float(np.max(np.abs(a))),
            float(np.max(np.abs(b))),
        )
        tolerance = 1e-9 * coordinate_scale

        a_min, a_max = np.min(a, axis=0), np.max(a, axis=0)
        b_min, b_max = np.min(b, axis=0), np.max(b, axis=0)
        bounding_boxes_overlap = bool(
            np.all(a_max >= b_min - tolerance) and np.all(b_max >= a_min - tolerance)
        )
        if bounding_boxes_overlap:
            if GeometryMixin._polygon_boundaries_intersect(a, b, tolerance):
                return 0.0

            # If boundaries do not cross, overlap can only occur by containment.
            if GeometryMixin._point_in_filled_polygon(a[0], b, tolerance) or (
                GeometryMixin._point_in_filled_polygon(b[0], a, tolerance)
            ):
                return 0.0

        distance = min(
            GeometryMixin._vertices_to_boundary_distance(a, b),
            GeometryMixin._vertices_to_boundary_distance(b, a),
        )
        return 0.0 if distance <= tolerance else float(distance)

    @staticmethod
    def _distance_polygon_points(polygon, name):
        points = np.asarray(polygon, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] < 3:
            raise ValueError(f"{name} must contain at least three 2-D points")
        points = points[:, :2]
        if not np.all(np.isfinite(points)):
            raise ValueError(f"{name} contains non-finite coordinates")
        if points.shape[0] > 3 and np.allclose(
            points[0], points[-1], rtol=0.0, atol=1e-12
        ):
            points = points[:-1]
        return points

    @staticmethod
    def _cross_2d(left, right):
        return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]

    @staticmethod
    def _polygon_boundaries_intersect(a, b, tolerance):
        """Vectorized segment-intersection test for two closed boundaries."""
        b_start = b
        b_end = np.roll(b, -1, axis=0)
        b_vec = b_end - b_start

        for a_start, a_end in zip(a, np.roll(a, -1, axis=0)):
            a_vec = a_end - a_start
            orient_b_start = GeometryMixin._cross_2d(a_vec, b_start - a_start)
            orient_b_end = GeometryMixin._cross_2d(a_vec, b_end - a_start)
            orient_a_start = GeometryMixin._cross_2d(b_vec, a_start - b_start)
            orient_a_end = GeometryMixin._cross_2d(b_vec, a_end - b_start)

            crosses_a = ((orient_b_start > tolerance) & (orient_b_end < -tolerance)) | (
                (orient_b_start < -tolerance) & (orient_b_end > tolerance)
            )
            crosses_b = ((orient_a_start > tolerance) & (orient_a_end < -tolerance)) | (
                (orient_a_start < -tolerance) & (orient_a_end > tolerance)
            )
            if np.any(crosses_a & crosses_b):
                return True

            # Include endpoint contact and collinear overlap.
            for point, orientation in (
                (b_start, orient_b_start),
                (b_end, orient_b_end),
            ):
                on_line = np.abs(orientation) <= tolerance
                in_x = (point[:, 0] >= min(a_start[0], a_end[0]) - tolerance) & (
                    point[:, 0] <= max(a_start[0], a_end[0]) + tolerance
                )
                in_y = (point[:, 1] >= min(a_start[1], a_end[1]) - tolerance) & (
                    point[:, 1] <= max(a_start[1], a_end[1]) + tolerance
                )
                if np.any(on_line & in_x & in_y):
                    return True

            a_start_on_b = (
                (np.abs(orient_a_start) <= tolerance)
                & (a_start[0] >= np.minimum(b_start[:, 0], b_end[:, 0]) - tolerance)
                & (a_start[0] <= np.maximum(b_start[:, 0], b_end[:, 0]) + tolerance)
                & (a_start[1] >= np.minimum(b_start[:, 1], b_end[:, 1]) - tolerance)
                & (a_start[1] <= np.maximum(b_start[:, 1], b_end[:, 1]) + tolerance)
            )
            a_end_on_b = (
                (np.abs(orient_a_end) <= tolerance)
                & (a_end[0] >= np.minimum(b_start[:, 0], b_end[:, 0]) - tolerance)
                & (a_end[0] <= np.maximum(b_start[:, 0], b_end[:, 0]) + tolerance)
                & (a_end[1] >= np.minimum(b_start[:, 1], b_end[:, 1]) - tolerance)
                & (a_end[1] <= np.maximum(b_start[:, 1], b_end[:, 1]) + tolerance)
            )
            if np.any(a_start_on_b | a_end_on_b):
                return True
        return False

    @staticmethod
    def _point_in_filled_polygon(point, polygon, tolerance):
        starts = polygon
        ends = np.roll(polygon, -1, axis=0)
        edge = ends - starts
        edge_length_sq = np.einsum("ij,ij->i", edge, edge)
        projection = np.zeros_like(edge_length_sq)
        nonzero = edge_length_sq > 0.0
        projection[nonzero] = (
            np.einsum("ij,ij->i", point - starts[nonzero], edge[nonzero])
            / edge_length_sq[nonzero]
        )
        projection = np.clip(projection, 0.0, 1.0)
        closest = starts + projection[:, None] * edge
        if float(np.min(np.linalg.norm(closest - point, axis=1))) <= tolerance:
            return True

        y = float(point[1])
        x = float(point[0])
        crosses_y = (starts[:, 1] > y) != (ends[:, 1] > y)
        safe_denominator = np.where(crosses_y, ends[:, 1] - starts[:, 1], 1.0)
        crossing_x = starts[:, 0] + (y - starts[:, 1]) * edge[:, 0] / safe_denominator
        return bool(np.count_nonzero(crosses_y & (x < crossing_x)) % 2)

    @staticmethod
    def _vertices_to_boundary_distance(vertices, boundary):
        starts = boundary
        edge = np.roll(boundary, -1, axis=0) - starts
        edge_length_sq = np.einsum("ij,ij->i", edge, edge)
        best = np.inf
        # Cap temporary arrays at roughly one million vertex-edge combinations.
        chunk_size = max(1, 1_000_000 // max(1, len(boundary)))
        for chunk_start in range(0, len(vertices), chunk_size):
            chunk = vertices[chunk_start : chunk_start + chunk_size]
            point_to_start = chunk[:, None, :] - starts[None, :, :]
            projection = np.divide(
                np.einsum("cbi,bi->cb", point_to_start, edge),
                edge_length_sq[None, :],
                out=np.zeros((len(chunk), len(boundary)), dtype=np.float64),
                where=edge_length_sq[None, :] > 0.0,
            )
            projection = np.clip(projection, 0.0, 1.0)
            residual = point_to_start - projection[:, :, None] * edge[None, :, :]
            best = min(
                best,
                float(np.sqrt(np.min(np.einsum("cbi,cbi->cb", residual, residual)))),
            )
        return best

    @staticmethod
    def _compute_droplet_dims_oriented(pts):
        """
        Compute droplet diameter/height from a fitted contact line on the original contour.

        The previous convex-hull heuristic often picked a chord on the dome rather than the
        actual contact line, which made the fitted rectangle drift. This version searches for
        the longest nearly straight contiguous contour segment that also acts as a supporting
        line for the rest of the droplet, then measures height as the farthest inward point.

        Returns:
            diameter (float)
            height (float)
            box_info (dict): {
                'baseline_p1': (x,y),
                'baseline_p2': (x,y),
                'apex_point': (x,y),
                'base_mid_point': (x,y),
                'corners': [(x,y), ...]
            }
        """
        if pts.shape[0] < 3:
            return 0.0, 0.0, None

        def fit_circle_kasa(x_vals, y_vals):
            """Kasa algebraic circle fit with proper data centering for numerical stability."""
            x = np.asarray(x_vals, dtype=np.float64)
            y = np.asarray(y_vals, dtype=np.float64)
            n = len(x)
            if n < 3:
                return None
            xm, ym = x.mean(), y.mean()
            u = x - xm
            v = y - ym
            Suu = np.dot(u, u)
            Svv = np.dot(v, v)
            Suv = np.dot(u, v)
            A = np.array([[Suu, Suv], [Suv, Svv]])
            b_vec = np.array(
                [
                    0.5 * (np.dot(u, u * u) + np.dot(u, v * v)),
                    0.5 * (np.dot(v, v * v) + np.dot(v, u * u)),
                ]
            )
            try:
                uc, vc = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                return None
            cx = xm + uc
            cy = ym + vc
            r_sq = uc * uc + vc * vc + (Suu + Svv) / n
            if r_sq <= 0:
                return None
            r = float(np.sqrt(r_sq))
            resid = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
            rms = float(np.sqrt(np.mean(resid**2))) if resid.size > 0 else 0.0
            return float(cx), float(cy), r, rms

        def fit_spherical_cap_1d(x_dome, y_dome, half_span):
            """
            Constrained spherical-cap fit for a sessile droplet.

            The center is fixed on the perpendicular bisector of the contact span
            (cx = 0 in local frame), so only cy is optimised.  The contact radius
            a = half_span fixes the sphere radius once cy is known:
                R = sqrt(a^2 + cy^2)
            Height of the cap above the baseline:
                h = cy + R

            This 1-D optimisation is much more robust than an unconstrained 3-D
            algebraic fit and is the physically correct model for a sessile droplet.

            Returns (cy, radius, rms) or None.
            """
            from scipy.optimize import minimize_scalar

            x = np.asarray(x_dome, dtype=np.float64)
            y = np.asarray(y_dome, dtype=np.float64)
            if len(x) < 3 or half_span <= 0:
                return None

            def cost(cy_val):
                r_val = np.sqrt(half_span * half_span + cy_val * cy_val)
                dists = np.sqrt(x * x + (y - cy_val) ** 2)
                return float(np.sum((dists - r_val) ** 2))

            # Initial estimate from apex height via spherical-cap geometry:
            # h = cy + R, R^2 = a^2 + cy^2  =>  cy = (h^2 - a^2) / (2*h)
            h_est = float(np.max(y)) if y.size > 0 else half_span
            a = half_span
            cy_init = (h_est * h_est - a * a) / (2.0 * h_est) if h_est > 1e-6 else 0.0
            lo = cy_init - 2.0 * a
            hi = cy_init + 2.0 * a

            try:
                opt = minimize_scalar(
                    cost,
                    bounds=(lo, hi),
                    method="bounded",
                    options={"xatol": 1e-4, "maxiter": 300},
                )
                cy = float(opt.x)
            except Exception:
                cy = cy_init

            r = float(np.sqrt(a * a + cy * cy))
            resid = np.sqrt(x * x + (y - cy) ** 2) - r
            rms = float(np.sqrt(np.mean(resid**2))) if resid.size > 0 else 0.0
            return cy, r, rms

        pts = np.asarray(pts, dtype=np.float64)
        n_pts = int(pts.shape[0])
        if n_pts < 3:
            return 0.0, 0.0, None

        bbox_min = np.min(pts, axis=0)
        bbox_max = np.max(pts, axis=0)
        diag = float(np.linalg.norm(bbox_max - bbox_min))
        if diag <= 1e-6:
            return 0.0, 0.0, None

        # ---- Convex-hull chord sweep for contact baseline ----
        #
        # The contact line of a sessile droplet is the longest chord that:
        #   (a) acts as an approximate supporting line (all points on one side), and
        #   (b) yields H <= D (physical constraint for any spherical cap).
        #
        # We iterate over all pairs of convex-hull vertices.  A typical hull has
        # 8-20 vertices, so this is O(n_hull^2) ~ a few hundred iterations at most.
        from scipy.spatial import ConvexHull

        try:
            hull = ConvexHull(pts)
            hull_verts = pts[hull.vertices]
        except Exception:
            hull_verts = pts

        n_hull = len(hull_verts)
        outside_tol = max(2.0, diag * 0.025)
        min_chord = max(8.0, diag * 0.15)

        best = None

        for i in range(n_hull):
            for j in range(n_hull):
                if i == j:
                    continue
                p1 = hull_verts[i]
                p2 = hull_verts[j]
                chord = p2 - p1
                chord_len = float(np.linalg.norm(chord))
                if chord_len < min_chord:
                    continue

                direction = chord / chord_len
                normal = np.array([-direction[1], direction[0]], dtype=np.float64)

                # Signed distances of all contour pts from the line through p1
                signed = np.dot(pts - p1, normal)

                # Orient normal so that the majority of pts are on the positive side
                if np.sum(signed > 0) < np.sum(signed < 0):
                    normal = -normal
                    signed = -signed

                # Reject if too many points lie on the wrong side of the line
                outside_frac = float(np.mean(signed < -outside_tol))
                if outside_frac > 0.05:
                    continue

                height = float(np.max(signed))
                if height <= 1.0:
                    continue

                # Score = chord length, with strong penalty when H > D
                hd = height / chord_len
                score = chord_len
                if hd > 1.0:
                    score *= 1.0 / (1.0 + (hd - 1.0) ** 2 * 50.0)

                if best is None or score > best["score"] + 1e-9:
                    best = {
                        "score": score,
                        "p1": p1,
                        "p2": p2,
                        "direction": direction,
                        "normal": normal,
                        "signed": signed,
                        "diameter": chord_len,
                        "height": height,
                    }

        # ---- Fallback: PCA orientation ----
        if best is None:
            centroid = np.mean(pts, axis=0)
            try:
                _u, _s, vh = np.linalg.svd(pts - centroid, full_matrices=False)
                direction = np.asarray(vh[0], dtype=np.float64)
                direction /= max(float(np.linalg.norm(direction)), 1e-9)
            except Exception:
                direction = np.array([1.0, 0.0], dtype=np.float64)
            normal = np.array([-direction[1], direction[0]], dtype=np.float64)
            signed = np.dot(pts - centroid, normal)
            if np.sum(signed > 0) < np.sum(signed < 0):
                normal = -normal
                signed = -signed
            u_all = np.dot(pts - centroid, direction)
            base_off = float(np.min(signed))
            signed -= base_off
            diameter = float(np.max(u_all) - np.min(u_all))
            height = float(np.max(signed))
            apex_world = pts[int(np.argmax(signed))]
            base_p1_world = (
                centroid + direction * float(np.min(u_all)) + normal * base_off
            )
            base_p2_world = (
                centroid + direction * float(np.max(u_all)) + normal * base_off
            )
            base_mid_world = 0.5 * (base_p1_world + base_p2_world)
            c1, c2 = base_p1_world, base_p2_world
            c3, c4 = c2 + normal * height, c1 + normal * height
            return (
                diameter,
                height,
                {
                    "baseline_p1": c1,
                    "baseline_p2": c2,
                    "apex_point": apex_world,
                    "base_mid_point": base_mid_world,
                    "corners": [c1, c2, c3, c4],
                    "arc_points": None,
                    "fit_center": None,
                    "fit_radius": None,
                },
            )

        # ---- Extract baseline from best hull chord ----
        base_p1_world = best["p1"]
        base_p2_world = best["p2"]
        direction = best["direction"]
        normal = best["normal"]
        signed_all = best["signed"]
        diameter = best["diameter"]
        height = best["height"]

        apex_idx = int(np.argmax(signed_all))
        apex_world = pts[apex_idx]
        base_mid_world = 0.5 * (base_p1_world + base_p2_world)

        c1, c2 = base_p1_world, base_p2_world
        c3, c4 = c2 + normal * height, c1 + normal * height

        # ---- Spherical-cap refinement on dome points ----
        fit_center_world = None
        fit_radius = None

        baseline_mid = base_mid_world
        local_x = np.dot(pts - baseline_mid, direction)
        local_y = np.dot(pts - baseline_mid, normal)
        dome_mask = local_y > max(0.5, height * 0.03)
        half_span = diameter / 2.0

        if int(np.sum(dome_mask)) >= 4 and half_span > 1e-6:
            x_dome = local_x[dome_mask]
            y_dome = local_y[dome_mask]

            # Primary: constrained spherical-cap (cx = 0)
            cap_fit = fit_spherical_cap_1d(x_dome, y_dome, half_span)
            used_constrained = False
            if cap_fit is not None:
                fit_cy, fit_r, fit_rms = cap_fit
                fit_height = float(fit_cy + fit_r)
                if fit_height > 1.0 and fit_rms <= max(5.0, height * 0.30):
                    height = float(fit_height)
                    apex_world = baseline_mid + normal * fit_height
                    c3 = c2 + normal * height
                    c4 = c1 + normal * height
                    fit_center_world = baseline_mid + normal * fit_cy
                    fit_radius = float(fit_r)
                    used_constrained = True

            # Fallback: Kasa unconstrained
            if not used_constrained:
                kasa = fit_circle_kasa(x_dome, y_dome)
                if kasa is not None:
                    fit_cx, fit_cy, fit_r, fit_rms = kasa
                    discriminant = fit_r * fit_r - fit_cy * fit_cy
                    if discriminant > 1.0 and fit_rms <= max(5.0, height * 0.30):
                        half_span_fit = float(np.sqrt(discriminant))
                        fit_height = float(fit_cy + fit_r)
                        if fit_height > 1.0:
                            diameter = float(2.0 * half_span_fit)
                            height = float(fit_height)
                            base_p1_world = baseline_mid + direction * float(
                                fit_cx - half_span_fit
                            )
                            base_p2_world = baseline_mid + direction * float(
                                fit_cx + half_span_fit
                            )
                            base_mid_world = baseline_mid + direction * float(fit_cx)
                            apex_world = base_mid_world + normal * fit_height
                            c1, c2 = base_p1_world, base_p2_world
                            c3, c4 = c2 + normal * height, c1 + normal * height
                            fit_center_world = (
                                baseline_mid
                                + direction * float(fit_cx)
                                + normal * float(fit_cy)
                            )
                            fit_radius = float(fit_r)

        return (
            diameter,
            height,
            {
                "baseline_p1": base_p1_world,
                "baseline_p2": base_p2_world,
                "apex_point": apex_world,
                "base_mid_point": base_mid_world,
                "corners": [c1, c2, c3, c4],
                "arc_points": None,
                "fit_center": fit_center_world,
                "fit_radius": fit_radius,
            },
        )

    @staticmethod
    def polygon_area(coords):
        """
        coords: (N,2) 不需要闭合
        """
        x = coords[:, 0]
        y = coords[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @staticmethod
    def polygon_centroid(coords):
        """Return the area centroid of a polygon and its absolute area."""
        pts = np.asarray(coords, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
            return None, 0.0

        pts = pts[:, :2]
        if pts.shape[0] < 3:
            return pts.mean(axis=0), 0.0

        x = pts[:, 0]
        y = pts[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        signed_area = 0.5 * np.sum(cross)
        if abs(signed_area) < 1e-12:
            return pts.mean(axis=0), 0.0

        cx = np.sum((x + x_next) * cross) / (6.0 * signed_area)
        cy = np.sum((y + y_next) * cross) / (6.0 * signed_area)
        return np.array([cx, cy], dtype=np.float64), abs(float(signed_area))
