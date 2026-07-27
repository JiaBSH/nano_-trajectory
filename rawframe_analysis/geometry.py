"""Polygon and droplet geometry calculations."""

from __future__ import annotations

import numpy as np


class GeometryMixin:
    """Provide polygon and droplet geometry calculations."""

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
