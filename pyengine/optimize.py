import numpy as np
from numba import njit

from .config import (FLIGHT_SPEED, HOURS_PER_ARRAY_SLICE, LEVEL_CHANGE_PENALTY,
                     LEVEL_PENALTIES, NUM_CANDIDATES, NUM_ITERATIONS,
                     OPTIMIZATION_RADIUS, POINTS_TO_ADJUST, SMOOTHNESS_WEIGHT)
from .danger import find_optimal_level, sample3d
from .geometry import level_distances, recalc_distances
from .rng import rk_double, rk_gauss


@njit(cache=True)
def line_clear(x1, y1, x2, y2, w, h):
    x1i = int(np.rint(x1))
    y1i = int(np.rint(y1))
    x2i = int(np.rint(x2))
    y2i = int(np.rint(y2))
    dx = abs(x2i - x1i)
    dy = abs(y2i - y1i)
    x = x1i
    y = y1i
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    if x2i < 0 or x2i >= w or y2i < 0 or y2i >= h:
        return False
    return True


@njit(cache=True)
def smooth_adjust(path_xs, path_ys, center_index, points_to_adjust, w, h, frozen):
    n_points = path_xs.shape[0]
    adj_xs = path_xs.copy()
    adj_ys = path_ys.copy()
    start_idx = max(1, center_index - points_to_adjust)
    end_idx = min(n_points - 2, center_index + points_to_adjust)
    for i in range(start_idx, end_idx + 1):
        if i == center_index:
            continue
        if frozen[i]:
            continue
        window_start = max(1, i - 1)
        window_end = min(n_points - 2, i + 1)
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
            sx = int(np.rint(smoothed_x))
            sy = int(np.rint(smoothed_y))
            if 0 <= sx < w and 0 <= sy < h:
                if line_clear(adj_xs[i - 1], adj_ys[i - 1], smoothed_x, smoothed_y, w, h):
                    if line_clear(smoothed_x, smoothed_y, adj_xs[i + 1], adj_ys[i + 1], w, h):
                        adj_xs[i] = smoothed_x
                        adj_ys[i] = smoothed_y
    return adj_xs, adj_ys


@njit(cache=True)
def is_candidate_safe(path_xs, path_ys, center_index, cand_x, cand_y,
                      points_to_adjust, w, h, frozen):
    n_points = path_xs.shape[0]
    temp_xs = path_xs.copy()
    temp_ys = path_ys.copy()
    temp_xs[center_index] = cand_x
    temp_ys[center_index] = cand_y
    temp_xs, temp_ys = smooth_adjust(temp_xs, temp_ys, center_index,
                                     points_to_adjust, w, h, frozen)
    start_idx = max(0, center_index - points_to_adjust)
    end_idx = min(n_points - 1, center_index + points_to_adjust)
    for i in range(start_idx, end_idx + 1):
        xi = int(np.rint(temp_xs[i]))
        yi = int(np.rint(temp_ys[i]))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            return False
    for i in range(start_idx, end_idx):
        if not line_clear(temp_xs[i], temp_ys[i], temp_xs[i + 1], temp_ys[i + 1], w, h):
            return False
    return True


@njit(cache=True)
def is_point_and_neighbors_safe(path_xs, path_ys, center_index, px, py, w, h):
    xi = int(np.rint(px))
    yi = int(np.rint(py))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return False
    if center_index > 0:
        if not line_clear(path_xs[center_index - 1], path_ys[center_index - 1], px, py, w, h):
            return False
    if center_index < path_xs.shape[0] - 1:
        if not line_clear(px, py, path_xs[center_index + 1], path_ys[center_index + 1], w, h):
            return False
    return True


@njit(cache=True)
def generate_candidates(cx0, cy0, num_candidates, max_attempts, radius, width, height,
                        use_direction, dir_x, dir_y, mt, idx, gauss):
    out_x = np.empty(num_candidates, dtype=np.float64)
    out_y = np.empty(num_candidates, dtype=np.float64)
    count = 0
    attempts = 0
    while count < num_candidates and attempts < max_attempts:
        attempts += 1
        take_dir = False
        if use_direction:
            if rk_double(mt, idx) < 0.7:
                take_dir = True
        if take_dir:
            angle_variation = 0.0 + 0.5 * rk_gauss(mt, idx, gauss)
            base_angle = np.arctan2(dir_y, dir_x)
            angle = base_angle + angle_variation
            r = rk_double(mt, idx) * radius
        else:
            angle = rk_double(mt, idx) * 2.0 * np.pi
            r = rk_double(mt, idx) * radius
        dx = r * np.cos(angle)
        dy = r * np.sin(angle)
        cx = cx0 + dx
        cy = cy0 + dy
        cx = max(0.0, min(width - 1.0, cx))
        cy = max(0.0, min(height - 1.0, cy))
        out_x[count] = cx
        out_y[count] = cy
        count += 1
    return out_x[:count], out_y[:count]


@njit(cache=True)
def path_fitness(xs, ys, levels, total_dists, danger2d, fc4, scale_x, scale_y,
                 w_length, w_safety, w_smooth, w_level_change, flight_speed, max_hours):
    n = xs.shape[0]
    if n < 2:
        return 1e18
    h2 = danger2d.shape[0]
    w2 = danger2d.shape[1]

    total_length = 0.0
    total_danger = 0.0
    total_curvature = 0.0
    total_level_changes = 0

    for i in range(n):
        xi = int(np.rint(xs[i]))
        yi = int(np.rint(ys[i]))
        if xi < 0 or xi >= w2 or yi < 0 or yi >= h2:
            return 1e18

    for i in range(n - 1):
        if not line_clear(xs[i], ys[i], xs[i + 1], ys[i + 1], w2, h2):
            return 1e18
        dx = xs[i + 1] - xs[i]
        dy = ys[i + 1] - ys[i]
        total_length += np.sqrt(dx * dx + dy * dy)

    for i in range(n):
        time_hours = total_dists[i] / flight_speed
        array_idx = int(time_hours)
        if array_idx < 0:
            array_idx = 0
        if array_idx >= max_hours:
            array_idx = max_hours - 1
        lvl = levels[i]
        level_danger = sample3d(fc4, array_idx, lvl, xs[i], ys[i], scale_x, scale_y)
        base_danger = danger2d[int(np.rint(ys[i])), int(np.rint(xs[i]))]
        total_danger += base_danger + level_danger + LEVEL_PENALTIES[lvl]

    for i in range(1, n - 1):
        dx1 = xs[i] - xs[i - 1]
        dy1 = ys[i] - ys[i - 1]
        dx2 = xs[i + 1] - xs[i]
        dy2 = ys[i + 1] - ys[i]
        norm1 = np.sqrt(dx1 * dx1 + dy1 * dy1)
        norm2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
        if norm1 > 0.0 and norm2 > 0.0:
            cos_angle = (dx1 * dx2 + dy1 * dy2) / (norm1 * norm2)
            if cos_angle > 1.0:
                cos_angle = 1.0
            if cos_angle < -1.0:
                cos_angle = -1.0
            sdx = xs[i + 1] - xs[i - 1]
            sdy = ys[i + 1] - ys[i - 1]
            span = np.sqrt(sdx * sdx + sdy * sdy)
            total_curvature += (1.0 - cos_angle) * span

    for i in range(1, n):
        if levels[i] != levels[i - 1]:
            total_level_changes += 1

    return (w_length * total_length + w_safety * total_danger +
            w_smooth * total_curvature + w_level_change * np.float64(total_level_changes))


def optimize_stage2(xs, ys, levels, stay, total, arrs, frozen, w_safety, w_length,
                    danger2d, fc4, scale_x, scale_y, mt, ridx, gauss,
                    dk_dang, dk_hour, order, ostate):
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
                use_direction, dir_x, dir_y, mt, ridx, gauss)

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
                cand_lv[c, i] = find_optimal_level(
                    cand_x[c], cand_y[c], current_level, current_level_distance,
                    cand_td[c, i], danger2d, fc4, scale_x, scale_y,
                    dk_dang, dk_hour, order, ostate)

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
