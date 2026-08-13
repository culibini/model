# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np

from libc.math cimport M_PI, atan2, cos, rint, sin, sqrt

from . import config as _cfg
from .danger import dist_to_hour
from .danger cimport sample3d
from .rng cimport RNG

cdef double LEVEL_CHANGE_PENALTY = _cfg.LEVEL_CHANGE_PENALTY
cdef double SMOOTHNESS_WEIGHT = _cfg.SMOOTHNESS_WEIGHT
cdef double FLIGHT_SPEED = _cfg.FLIGHT_SPEED
cdef double HOURS_PER_ARRAY_SLICE = _cfg.HOURS_PER_ARRAY_SLICE
cdef double OPTIMIZATION_RADIUS = _cfg.OPTIMIZATION_RADIUS
cdef long long NUM_CANDIDATES = _cfg.NUM_CANDIDATES
cdef long long NUM_ITERATIONS = _cfg.NUM_ITERATIONS
cdef long long POINTS_TO_ADJUST = _cfg.POINTS_TO_ADJUST

_LEVEL_PENALTIES = np.ascontiguousarray(_cfg.LEVEL_PENALTIES, dtype=np.float64)
cdef double[::1] LEVEL_PENALTIES = _LEVEL_PENALTIES


cdef bint line_clear(double x1, double y1, double x2, double y2,
                     long long w, long long h) noexcept nogil:
    cdef long long x1i = <long long>rint(x1)
    cdef long long y1i = <long long>rint(y1)
    cdef long long x2i = <long long>rint(x2)
    cdef long long y2i = <long long>rint(y2)
    if x1i < 0 or x1i >= w or y1i < 0 or y1i >= h:
        return False
    if x2i < 0 or x2i >= w or y2i < 0 or y2i >= h:
        return False
    return True


cdef void _smooth_adjust_inplace(double[::1] adj_xs, double[::1] adj_ys,
                                 long long center_index, long long points_to_adjust,
                                 long long w, long long h,
                                 const unsigned char[::1] frozen) noexcept nogil:
    cdef long long n_points = adj_xs.shape[0]
    cdef long long start_idx = center_index - points_to_adjust
    cdef long long end_idx = center_index + points_to_adjust
    cdef long long i, j, window_start, window_end, count, sx, sy
    cdef double sum_x, sum_y, smoothed_x, smoothed_y
    if start_idx < 1:
        start_idx = 1
    if end_idx > n_points - 2:
        end_idx = n_points - 2
    for i in range(start_idx, end_idx + 1):
        if i == center_index:
            continue
        if frozen[i]:
            continue
        window_start = i - 1 if i - 1 > 1 else 1
        window_end = i + 1 if i + 1 < n_points - 2 else n_points - 2
        count = 0
        sum_x = 0.0
        sum_y = 0.0
        for j in range(window_start, window_end + 1):
            sum_x += adj_xs[j]
            sum_y += adj_ys[j]
            count += 1
        if count > 0:
            smoothed_x = sum_x / count
            smoothed_y = sum_y / count
            sx = <long long>rint(smoothed_x)
            sy = <long long>rint(smoothed_y)
            if 0 <= sx < w and 0 <= sy < h:
                if line_clear(adj_xs[i - 1], adj_ys[i - 1], smoothed_x, smoothed_y, w, h):
                    if line_clear(smoothed_x, smoothed_y, adj_xs[i + 1], adj_ys[i + 1], w, h):
                        adj_xs[i] = smoothed_x
                        adj_ys[i] = smoothed_y


def smooth_adjust(path_xs, path_ys, long long center_index, long long points_to_adjust,
                  long long w, long long h, frozen):
    adj_xs = path_xs.copy()
    adj_ys = path_ys.copy()
    _smooth_adjust_inplace(adj_xs, adj_ys, center_index, points_to_adjust, w, h, frozen)
    return adj_xs, adj_ys


def is_candidate_safe(path_xs, path_ys, long long center_index,
                      double cand_x, double cand_y, long long points_to_adjust,
                      long long w, long long h, frozen):
    cdef long long n_points = path_xs.shape[0]
    temp_xs = path_xs.copy()
    temp_ys = path_ys.copy()
    temp_xs[center_index] = cand_x
    temp_ys[center_index] = cand_y
    _smooth_adjust_inplace(temp_xs, temp_ys, center_index, points_to_adjust, w, h, frozen)
    cdef double[::1] txs = temp_xs
    cdef double[::1] tys = temp_ys
    cdef long long start_idx = center_index - points_to_adjust
    cdef long long end_idx = center_index + points_to_adjust
    cdef long long i, xi, yi
    if start_idx < 0:
        start_idx = 0
    if end_idx > n_points - 1:
        end_idx = n_points - 1
    for i in range(start_idx, end_idx + 1):
        xi = <long long>rint(txs[i])
        yi = <long long>rint(tys[i])
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            return False
    for i in range(start_idx, end_idx):
        if not line_clear(txs[i], tys[i], txs[i + 1], tys[i + 1], w, h):
            return False
    return True


def is_point_and_neighbors_safe(path_xs, path_ys, long long center_index,
                                double px, double py, long long w, long long h):
    cdef double[::1] xs = path_xs
    cdef double[::1] ys = path_ys
    cdef long long xi = <long long>rint(px)
    cdef long long yi = <long long>rint(py)
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return False
    if center_index > 0:
        if not line_clear(xs[center_index - 1], ys[center_index - 1], px, py, w, h):
            return False
    if center_index < xs.shape[0] - 1:
        if not line_clear(px, py, xs[center_index + 1], ys[center_index + 1], w, h):
            return False
    return True


def generate_candidates(double cx0, double cy0, long long num_candidates,
                        long long max_attempts, double radius,
                        double width, double height,
                        bint use_direction, double dir_x, double dir_y, RNG rng):
    out_x = np.empty(num_candidates, dtype=np.float64)
    out_y = np.empty(num_candidates, dtype=np.float64)
    cdef double[::1] ox = out_x
    cdef double[::1] oy = out_y
    cdef long long count = 0
    cdef long long attempts = 0
    cdef bint take_dir
    cdef double angle_variation, base_angle, angle, r, dx, dy, cx, cy
    while count < num_candidates and attempts < max_attempts:
        attempts += 1
        take_dir = False
        if use_direction:
            if rng.rk_double() < 0.7:
                take_dir = True
        if take_dir:
            angle_variation = 0.0 + 0.5 * rng.rk_gauss()
            base_angle = atan2(dir_y, dir_x)
            angle = base_angle + angle_variation
            r = rng.rk_double() * radius
        else:
            angle = rng.rk_double() * 2.0 * M_PI
            r = rng.rk_double() * radius
        dx = r * cos(angle)
        dy = r * sin(angle)
        cx = cx0 + dx
        cy = cy0 + dy
        if cx > width - 1.0:
            cx = width - 1.0
        if cx < 0.0:
            cx = 0.0
        if cy > height - 1.0:
            cy = height - 1.0
        if cy < 0.0:
            cy = 0.0
        ox[count] = cx
        oy[count] = cy
        count += 1
    return out_x[:count], out_y[:count]


def recalc_distances(xs, ys):
    cdef double[::1] x = xs
    cdef double[::1] y = ys
    cdef long long n = x.shape[0]
    out_arr = np.zeros(n, dtype=np.float64)
    cdef double[::1] out = out_arr
    cdef long long i
    cdef double dx, dy
    for i in range(1, n):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        out[i] = out[i - 1] + sqrt(dx * dx + dy * dy)
    return out_arr


def level_distances(xs, ys, levels):
    cdef double[::1] x = xs
    cdef double[::1] y = ys
    cdef long long[::1] lv = levels
    cdef long long n = x.shape[0]
    out_arr = np.zeros(n, dtype=np.float64)
    cdef double[::1] out = out_arr
    cdef long long i
    cdef double dx, dy
    for i in range(1, n):
        if lv[i] == lv[i - 1]:
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
            out[i] = out[i - 1] + sqrt(dx * dx + dy * dy)
        else:
            out[i] = 0.0
    return out_arr


def path_fitness(xs, ys, levels, total_dists, danger2d, fc4,
                 double scale_x, double scale_y,
                 double w_length, double w_safety, double w_smooth,
                 double w_level_change, double flight_speed, long long max_hours):
    cdef double[::1] x = xs
    cdef double[::1] y = ys
    cdef long long[::1] lv = levels
    cdef double[::1] td = total_dists
    cdef double[:, ::1] d2 = danger2d
    cdef const double[:, :, :, ::1] f4 = fc4
    cdef long long n = x.shape[0]
    if n < 2:
        return 1e18
    cdef long long h2 = d2.shape[0]
    cdef long long w2 = d2.shape[1]

    cdef double total_length = 0.0
    cdef double total_danger = 0.0
    cdef double total_curvature = 0.0
    cdef long long total_level_changes = 0
    cdef long long i, xi, yi, array_idx, lvl
    cdef double dx, dy, time_hours, level_danger, base_danger
    cdef double dx1, dy1, dx2, dy2, norm1, norm2, cos_angle, sdx, sdy, span

    for i in range(n):
        xi = <long long>rint(x[i])
        yi = <long long>rint(y[i])
        if xi < 0 or xi >= w2 or yi < 0 or yi >= h2:
            return 1e18

    for i in range(n - 1):
        if not line_clear(x[i], y[i], x[i + 1], y[i + 1], w2, h2):
            return 1e18
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        total_length += sqrt(dx * dx + dy * dy)

    for i in range(n):
        time_hours = td[i] / flight_speed
        array_idx = <long long>time_hours
        if array_idx < 0:
            array_idx = 0
        if array_idx >= max_hours:
            array_idx = max_hours - 1
        lvl = lv[i]
        level_danger = sample3d(f4, array_idx, lvl, x[i], y[i], scale_x, scale_y)
        base_danger = d2[<long long>rint(y[i]), <long long>rint(x[i])]
        total_danger += base_danger + level_danger + LEVEL_PENALTIES[lvl]

    for i in range(1, n - 1):
        dx1 = x[i] - x[i - 1]
        dy1 = y[i] - y[i - 1]
        dx2 = x[i + 1] - x[i]
        dy2 = y[i + 1] - y[i]
        norm1 = sqrt(dx1 * dx1 + dy1 * dy1)
        norm2 = sqrt(dx2 * dx2 + dy2 * dy2)
        if norm1 > 0.0 and norm2 > 0.0:
            cos_angle = (dx1 * dx2 + dy1 * dy2) / (norm1 * norm2)
            if cos_angle > 1.0:
                cos_angle = 1.0
            if cos_angle < -1.0:
                cos_angle = -1.0
            sdx = x[i + 1] - x[i - 1]
            sdy = y[i + 1] - y[i - 1]
            span = sqrt(sdx * sdx + sdy * sdy)
            total_curvature += (1.0 - cos_angle) * span

    for i in range(1, n):
        if lv[i] != lv[i - 1]:
            total_level_changes += 1

    return (w_length * total_length + w_safety * total_danger +
            w_smooth * total_curvature + w_level_change * <double>total_level_changes)


def optimize_stage2(xs, ys, levels, stay, total, arrs, frozen, w_safety, w_length,
                    danger2d, fc4, scale_x, scale_y, rng, cache):
    n_points = len(xs)
    path_xs = np.array(xs, dtype=np.float64)
    path_ys = np.array(ys, dtype=np.float64)
    path_levels = np.array(levels, dtype=np.int64)
    path_total = np.array(total, dtype=np.float64)
    frozen = np.asarray(frozen, dtype=np.uint8)
    if frozen.shape[0] != n_points:
        frozen = np.zeros(n_points, dtype=np.uint8)

    h2, w2 = danger2d.shape
    n_hours = fc4.shape[0]
    fs_scaled = FLIGHT_SPEED * HOURS_PER_ARRAY_SLICE

    lvl_dist = level_distances(path_xs, path_ys, path_levels)

    def fit(pxs, pys, plv, ptd):
        return path_fitness(pxs, pys, plv, ptd, danger2d, fc4, scale_x, scale_y,
                            w_length, w_safety, SMOOTHNESS_WEIGHT, LEVEL_CHANGE_PENALTY,
                            fs_scaled, n_hours)

    best_seen = fit(path_xs, path_ys, path_levels, path_total)
    best_xs = path_xs.copy()
    best_ys = path_ys.copy()
    best_td = path_total.copy()
    best_lv = path_levels.copy()

    for _ in range(NUM_ITERATIONS):
        for i in range(1, n_points - 1):
            if frozen[i]:
                continue

            best_fitness = fit(path_xs, path_ys, path_levels, path_total)

            cur_x = path_xs[i]
            cur_y = path_ys[i]
            current_level = int(path_levels[i])
            current_level_distance = lvl_dist[i]

            best_cx, best_cy, best_level = cur_x, cur_y, current_level

            use_direction = False
            dir_x = dir_y = 0.0
            if 0 < i < n_points - 1:
                ideal_x = (path_xs[i - 1] + path_xs[i + 1]) / 2.0
                ideal_y = (path_ys[i - 1] + path_ys[i + 1]) / 2.0
                ddx = ideal_x - cur_x
                ddy = ideal_y - cur_y
                dist_to_ideal = float(np.sqrt(ddx * ddx + ddy * ddy))
                if dist_to_ideal > 0.0:
                    dir_x = ddx / dist_to_ideal
                    dir_y = ddy / dist_to_ideal
                    use_direction = True

            gx_arr, gy_arr = generate_candidates(
                cur_x, cur_y, NUM_CANDIDATES, NUM_CANDIDATES * 10,
                OPTIMIZATION_RADIUS, float(w2), float(h2),
                use_direction, dir_x, dir_y, rng)

            cand_x = []
            cand_y = []
            for c in range(gx_arr.shape[0]):
                if is_candidate_safe(path_xs, path_ys, i, gx_arr[c], gy_arr[c],
                                     POINTS_TO_ADJUST, w2, h2, frozen):
                    cand_x.append(gx_arr[c])
                    cand_y.append(gy_arr[c])
            if is_point_and_neighbors_safe(path_xs, path_ys, i, cur_x, cur_y, w2, h2):
                cand_x.append(cur_x)
                cand_y.append(cur_y)
            if not cand_x:
                continue

            ncand = len(cand_x)
            cand_tx = np.empty((ncand, n_points), dtype=np.float64)
            cand_ty = np.empty((ncand, n_points), dtype=np.float64)
            cand_td = np.empty((ncand, n_points), dtype=np.float64)
            cand_lv = np.empty((ncand, n_points), dtype=np.int64)

            for c in range(ncand):
                tx = path_xs.copy()
                ty = path_ys.copy()
                tx[i] = cand_x[c]
                ty[i] = cand_y[c]
                sx_a, sy_a = smooth_adjust(tx, ty, i, POINTS_TO_ADJUST, w2, h2, frozen)
                cand_tx[c] = sx_a
                cand_ty[c] = sy_a
                cand_td[c] = recalc_distances(sx_a, sy_a)
                cand_lv[c] = path_levels
                cand_lv[c, i] = cache.find_optimal_level(
                    cand_x[c], cand_y[c], current_level, current_level_distance,
                    cand_td[c, i])

            for c in range(ncand):
                f = fit(cand_tx[c], cand_ty[c], cand_lv[c], cand_td[c])
                if f < best_fitness:
                    best_fitness = f
                    best_cx = cand_x[c]
                    best_cy = cand_y[c]
                    best_level = int(cand_lv[c, i])

            path_xs[i] = best_cx
            path_ys[i] = best_cy
            path_levels[i] = best_level

            path_xs, path_ys = smooth_adjust(path_xs, path_ys, i,
                                             POINTS_TO_ADJUST, w2, h2, frozen)
            path_total = recalc_distances(path_xs, path_ys)
            lvl_dist = level_distances(path_xs, path_ys, path_levels)

        current_fitness = fit(path_xs, path_ys, path_levels, path_total)
        if current_fitness < best_seen:
            best_seen = current_fitness
            best_xs = path_xs.copy()
            best_ys = path_ys.copy()
            best_td = path_total.copy()
            best_lv = path_levels.copy()

    return best_xs, best_ys, best_lv, best_td
