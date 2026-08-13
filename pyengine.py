import sys
import time

import numpy as np
from numba import njit, prange, set_num_threads
from numba import types
from numba.typed import Dict

LENGTH_WEIGHT = 1.0
SAFETY_WEIGHT = 5.0
SMOOTHNESS_WEIGHT = 12.0
LEVEL_CHANGE_PENALTY = 0.5
SAFETY_WEIGHT_COEFF = 0.5
BASE_PATH_LENGTH = 4000.0
OPTIMIZATION_RADIUS = 400.0
POINTS_TO_ADJUST = 5
NUM_CANDIDATES = 60
NUM_ITERATIONS = 3
LEVEL_OPTIMIZATION_RANGE = 3
LOOKAHEAD_LEVELS = 10
LEVEL_STAY_MULTIPLIER = 10.0
NUM_LEVELS = 32
DIJKSTRA_STEP_SIZE = 20
DIJKSTRA_LENGTH_PENALTY = 0.25
DIJKSTRA_DANGER_WEIGHT = 2.0
DIJKSTRA_GOAL_TOLERANCE = 20.0
DANGER_SOFT_LIMIT = 240.0
DANGER_HYPERBOLIC_RANGE = 60.0
DANGER_HYPERBOLIC_SCALE = 1000.0
FLIGHT_SPEED = 1000.0
HOURS_PER_ARRAY_SLICE = 1.0
HOUR_STAY_DISTANCE = FLIGHT_SPEED * 0.25
HOUR_SWITCH_PENALTY = 25.0
MAX_NODES = 35000000
RNG_SEED = 12345

LEVEL_PENALTIES = np.array([
    10, 10, 5, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 5,
    10, 10, 20, 20, 30, 30, 40, 40, 50, 50,
    60, 60], dtype=np.float64)


@njit(cache=True)
def rk_next(mt, idx):
    if idx[0] >= 624:
        for i in range(624):
            y = (mt[i] & np.uint32(0x80000000)) | (mt[(i + 1) % 624] & np.uint32(0x7FFFFFFF))
            v = mt[(i + 397) % 624] ^ (y >> np.uint32(1))
            if y & np.uint32(1):
                v ^= np.uint32(0x9908B0DF)
            mt[i] = v
        idx[0] = 0
    y = mt[idx[0]]
    idx[0] += 1
    y ^= y >> np.uint32(11)
    y ^= (y << np.uint32(7)) & np.uint32(0x9D2C5680)
    y ^= (y << np.uint32(15)) & np.uint32(0xEFC60000)
    y ^= y >> np.uint32(18)
    return y


@njit(cache=True)
def rng_seed(mt, idx, gauss, seed):
    mt[0] = np.uint32(seed)
    for i in range(1, 624):
        mt[i] = np.uint32(np.uint32(1812433253) * (mt[i - 1] ^ (mt[i - 1] >> np.uint32(30))) + np.uint32(i))
    idx[0] = 624
    gauss[0] = 0.0
    gauss[1] = 0.0


@njit(cache=True)
def rk_double(mt, idx):
    a = np.float64(rk_next(mt, idx) >> np.uint32(5))
    b = np.float64(rk_next(mt, idx) >> np.uint32(6))
    return (a * 67108864.0 + b) / 9007199254740992.0


@njit(cache=True)
def rk_gauss(mt, idx, gauss):
    if gauss[0] != 0.0:
        t = gauss[1]
        gauss[0] = 0.0
        gauss[1] = 0.0
        return t
    while True:
        x1 = 2.0 * rk_double(mt, idx) - 1.0
        x2 = 2.0 * rk_double(mt, idx) - 1.0
        r2 = x1 * x1 + x2 * x2
        if r2 < 1.0 and r2 != 0.0:
            break
    f = np.sqrt(-2.0 * np.log(r2) / r2)
    gauss[0] = 1.0
    gauss[1] = f * x1
    return f * x2


@njit(cache=True)
def hyperbolic_danger(v):
    if DANGER_HYPERBOLIC_RANGE <= 0.0:
        return v
    delta = v - DANGER_SOFT_LIMIT
    if delta <= 0.0:
        return v
    cap = DANGER_HYPERBOLIC_RANGE * 0.999999
    capped = min(delta, cap)
    denom = max(1e-6, DANGER_HYPERBOLIC_RANGE - capped)
    return v + DANGER_HYPERBOLIC_SCALE * (capped / denom)


@njit(cache=True, parallel=True)
def apply_hyperbolic(a):
    flat = a.reshape(-1)
    for i in prange(flat.shape[0]):
        flat[i] = hyperbolic_danger(flat[i])


@njit(cache=True)
def sample3d(fc4, hour, level, px, py, scale_x, scale_y):
    L = fc4.shape[1]
    h3 = fc4.shape[2]
    w3 = fc4.shape[3]
    if level < 0 or level >= L:
        return 0.0
    u = px / scale_x - 0.5
    v = py / scale_y - 0.5
    if u < 0.0:
        u = 0.0
    if v < 0.0:
        v = 0.0
    mu = np.float64(w3) - 1.0
    mv = np.float64(h3) - 1.0
    if u > mu:
        u = mu
    if v > mv:
        v = mv
    x0 = int(u)
    y0 = int(v)
    x1 = min(x0 + 1, w3 - 1)
    y1 = min(y0 + 1, h3 - 1)
    fu = u - x0
    fv = v - y0
    d00 = fc4[hour, level, y0, x0]
    d10 = fc4[hour, level, y0, x1]
    d01 = fc4[hour, level, y1, x0]
    d11 = fc4[hour, level, y1, x1]
    return (d00 * (1.0 - fu) + d10 * fu) * (1.0 - fv) + (d01 * (1.0 - fu) + d11 * fu) * fv


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
def recalc_distances(xs, ys):
    n = xs.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        out[i] = out[i - 1] + np.sqrt(dx * dx + dy * dy)
    return out


@njit(cache=True)
def level_distances(xs, ys, levels):
    n = xs.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if levels[i] == levels[i - 1]:
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            out[i] = out[i - 1] + np.sqrt(dx * dx + dy * dy)
        else:
            out[i] = 0.0
    return out


@njit(cache=True)
def dist_to_hour(dist, arrays_count):
    time_hours = dist / FLIGHT_SPEED
    hour_idx = int(time_hours / HOURS_PER_ARRAY_SLICE)
    if hour_idx < 0:
        return 0
    max_idx = arrays_count - 1
    return hour_idx if hour_idx <= max_idx else max_idx


@njit(cache=True)
def cache_total_danger(px, py, level, dist, danger2d, fc4, scale_x, scale_y,
                       dk_dang, dk_hour, order, ostate):
    x = np.int64(np.rint(px))
    y = np.int64(np.rint(py))
    n_hours = fc4.shape[0]
    hour_idx = dist_to_hour(dist, n_hours)
    key = np.uint64((np.uint64(x) << np.uint64(40)) | (np.uint64(y) << np.uint64(20)) |
                    (np.uint64(level) << np.uint64(8)) | np.uint64(hour_idx))
    if key in dk_dang:
        return dk_dang[key], dk_hour[key]
    h2 = danger2d.shape[0]
    w2 = danger2d.shape[1]
    yi = int(np.rint(py))
    xi = int(np.rint(px))
    if xi < 0 or xi >= w2 or yi < 0 or yi >= h2:
        base = 1e9
    else:
        base = danger2d[yi, xi]
    penalty = LEVEL_PENALTIES[level] if 0 <= level < NUM_LEVELS else 0.0
    level_danger = 0.0
    if 0 <= hour_idx < n_hours:
        level_danger = sample3d(fc4, hour_idx, level, px, py, scale_x, scale_y)
    total = base + level_danger + penalty
    dk_dang[key] = total
    dk_hour[key] = np.int64(hour_idx)
    if ostate[1] >= order.shape[0]:
        keep = ostate[1] - ostate[0]
        for i in range(keep):
            order[i] = order[ostate[0] + i]
        ostate[1] = keep
        ostate[0] = 0
    order[ostate[1]] = key
    ostate[1] += 1
    if len(dk_dang) > 500000:
        to_drop = min(50000, ostate[1] - ostate[0])
        for i in range(to_drop):
            k = order[ostate[0] + i]
            if k in dk_dang:
                del dk_dang[k]
                del dk_hour[k]
        ostate[0] += to_drop
        if ostate[0] > ostate[1] // 2:
            keep = ostate[1] - ostate[0]
            for i in range(keep):
                order[i] = order[ostate[0] + i]
            ostate[1] = keep
            ostate[0] = 0
    return total, np.int64(hour_idx)


@njit(cache=True)
def find_best_level(px, py, dist, danger2d, fc4, scale_x, scale_y,
                    dk_dang, dk_hour, order, ostate):
    best_level = 0
    best_danger = np.inf
    for level in range(NUM_LEVELS):
        d, _ = cache_total_danger(px, py, level, dist, danger2d, fc4,
                                  scale_x, scale_y, dk_dang, dk_hour, order, ostate)
        if d < best_danger:
            best_danger = d
            best_level = level
    return best_level


@njit(cache=True)
def find_optimal_level(px, py, current_level, current_level_distance, total_distance,
                       danger2d, fc4, scale_x, scale_y, dk_dang, dk_hour, order, ostate):
    best_level = current_level
    bd, bh = cache_total_danger(px, py, current_level, total_distance, danger2d, fc4,
                                scale_x, scale_y, dk_dang, dk_hour, order, ostate)
    for off in range(-LEVEL_OPTIMIZATION_RANGE, LEVEL_OPTIMIZATION_RANGE + 1):
        cand = current_level + off
        if cand < 0 or cand >= NUM_LEVELS:
            continue
        if cand > current_level:
            if current_level_distance < LEVEL_STAY_MULTIPLIER * (cand - current_level):
                continue
        d, h = cache_total_danger(px, py, cand, total_distance, danger2d, fc4,
                                  scale_x, scale_y, dk_dang, dk_hour, order, ostate)
        if d < bd or (d == bd and h < bh):
            bd = d
            bh = h
            best_level = cand
    return best_level


@njit(cache=True, parallel=True)
def build_danger_grid(danger2d, fc4, penalties, grid_w, grid_h, step_size,
                      num_levels, max_hours, width, height, scale_x, scale_y):
    dg = np.zeros((grid_h, grid_w, num_levels, max_hours), dtype=np.float64)
    for gy in prange(grid_h):
        for gx in range(grid_w):
            nx_real = gx * step_size
            ny_real = gy * step_size
            if nx_real >= width or ny_real >= height:
                continue
            base = danger2d[ny_real, nx_real]
            for level in range(num_levels):
                lp = penalties[level]
                for hour in range(max_hours):
                    ld = sample3d(fc4, hour, level, np.float64(nx_real),
                                  np.float64(ny_real), scale_x, scale_y)
                    dg[gy, gx, level, hour] = base + ld + lp
    return dg


@njit(cache=True, parallel=True)
def build_min_nb(dg, grid_w, grid_h, step_size, width, height, num_levels, max_hours):
    mn = np.zeros((grid_h, grid_w, num_levels, max_hours), dtype=np.float64)
    has_nb = np.zeros(grid_h * grid_w, dtype=np.uint8)
    dxs = np.array([0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
    dys = np.array([1, 0, -1, 0, 1, -1, 1, -1], dtype=np.int64)
    for gy in prange(grid_h):
        for gx in range(grid_w):
            first = True
            for b in range(8):
                bx = gx + dxs[b]
                by = gy + dys[b]
                if bx < 0 or by < 0 or bx >= grid_w or by >= grid_h:
                    continue
                bxr = bx * step_size
                byr = by * step_size
                if bxr < 0 or byr < 0 or bxr >= width or byr >= height:
                    continue
                if first:
                    has_nb[gy * grid_w + gx] = 1
                    for l in range(num_levels):
                        for h in range(max_hours):
                            mn[gy, gx, l, h] = dg[by, bx, l, h]
                    first = False
                else:
                    for l in range(num_levels):
                        for h in range(max_hours):
                            v = dg[by, bx, l, h]
                            if v < mn[gy, gx, l, h]:
                                mn[gy, gx, l, h] = v
    return mn, has_nb


@njit(cache=True, parallel=True)
def build_min_dh(dg, grid_w, grid_h, num_levels, max_hours):
    md = np.empty(grid_h * grid_w, dtype=np.float64)
    for gy in prange(grid_h):
        for gx in range(grid_w):
            m = dg[gy, gx, 0, 0]
            for l in range(num_levels):
                for h in range(max_hours):
                    v = dg[gy, gx, l, h]
                    if v < m:
                        m = v
            md[gy * grid_w + gx] = m
    return md


@njit(cache=True)
def build_hgrid(md, grid_w, grid_h, step_size, width, height,
                goal_x, goal_y, goal_tolerance, danger_weight, length_penalty):
    n2 = grid_h * grid_w
    hg = np.full(n2, 1e18, dtype=np.float64)
    cap = 4 * n2 + 16
    hp = np.empty(cap, dtype=np.float64)
    hc = np.empty(cap, dtype=np.int64)
    hs = 0
    for gy in range(grid_h):
        for gx in range(grid_w):
            rx = np.float64(gx * step_size)
            ry = np.float64(gy * step_size)
            if rx >= width or ry >= height:
                continue
            if abs(rx - goal_x) <= goal_tolerance and abs(ry - goal_y) <= goal_tolerance:
                c = gy * grid_w + gx
                hg[c] = 0.0
                i = hs
                hp[i] = 0.0
                hc[i] = c
                while i > 0:
                    p = (i - 1) // 2
                    if hp[p] <= hp[i]:
                        break
                    hp[p], hp[i] = hp[i], hp[p]
                    hc[p], hc[i] = hc[i], hc[p]
                    i = p
                hs += 1
    s2 = np.sqrt(2.0)
    dxs = np.array([0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
    dys = np.array([1, 0, -1, 0, 1, -1, 1, -1], dtype=np.int64)
    while hs > 0:
        top_p = hp[0]
        top_c = hc[0]
        hs -= 1
        hp[0] = hp[hs]
        hc[0] = hc[hs]
        i = 0
        while True:
            l = 2 * i + 1
            r = 2 * i + 2
            sm = i
            if l < hs and hp[l] < hp[sm]:
                sm = l
            if r < hs and hp[r] < hp[sm]:
                sm = r
            if sm == i:
                break
            hp[i], hp[sm] = hp[sm], hp[i]
            hc[i], hc[sm] = hc[sm], hc[i]
            i = sm
        if top_p > hg[top_c]:
            continue
        cgx = top_c % grid_w
        cgy = top_c // grid_w
        enter_c = md[top_c] * danger_weight
        for d in range(8):
            agx = cgx + dxs[d]
            agy = cgy + dys[d]
            if agx < 0 or agy < 0 or agx >= grid_w or agy >= grid_h:
                continue
            arx = agx * step_size
            ary = agy * step_size
            if arx >= width or ary >= height:
                continue
            mul = 1.0 if d < 4 else s2
            cand = top_p + enter_c + mul * step_size * length_penalty
            ai = agy * grid_w + agx
            if cand < hg[ai]:
                hg[ai] = cand
                if hs >= cap:
                    continue
                i = hs
                hp[i] = cand
                hc[i] = ai
                while i > 0:
                    p = (i - 1) // 2
                    if hp[p] <= hp[i]:
                        break
                    hp[p], hp[i] = hp[i], hp[p]
                    hc[p], hc[i] = hc[i], hc[p]
                    i = p
                hs += 1
    return hg


@njit(cache=True)
def kernel(start_x, start_y, goal_x, goal_y, start_level, goal_level,
           danger2d, dg, mn, has_nb, hgrid, grid_w, grid_h, width, height,
           step_size, lookahead_levels, level_stay_multiplier, num_levels,
           flight_speed, max_hours, danger_weight, length_penalty,
           goal_tolerance, max_nodes, hour_lookahead, hour_stay_distance,
           hour_switch_penalty, wastar):
    n4 = grid_h * grid_w * num_levels * max_hours
    dgf = dg.reshape(-1)
    mnf = mn.reshape(-1)
    block = num_levels * max_hours

    start_gx = int(np.rint(start_x / np.float64(step_size)))
    start_gy = int(np.rint(start_y / np.float64(step_size)))

    inf = 1e18
    dist_grid = np.full(n4, inf, dtype=np.float64)
    aux_stay = np.zeros(n4, dtype=np.float64)
    aux_hstay = np.zeros(n4, dtype=np.float64)
    aux_tdist = np.zeros(n4, dtype=np.float64)
    NO_PREV = np.uint32(0xFFFFFFFF)
    prev_state = np.full(n4, NO_PREV, dtype=np.uint32)
    visited = np.zeros(n4, dtype=np.uint8)

    cap = min(max_nodes, n4) + 10
    hp = np.empty(cap, dtype=np.float64)
    hcost = np.empty(cap, dtype=np.float64)
    hstate = np.empty(cap, dtype=np.uint32)
    hs = 0

    start_ni = ((start_gy * grid_w + start_gx) * num_levels + start_level) * max_hours + 0
    dist_grid[start_ni] = 0.0
    hp[0] = 0.0
    hcost[0] = 0.0
    hstate[0] = np.uint32(start_ni)
    hs = 1

    s2 = np.sqrt(2.0)
    dir_dx = np.array([0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
    dir_dy = np.array([1, 0, -1, 0, 1, -1, 1, -1], dtype=np.int64)
    dir_mul = np.array([1.0, 1.0, 1.0, 1.0, s2, s2, s2, s2], dtype=np.float64)

    nb_dir = np.empty(8, dtype=np.int64)
    nb_block = np.empty(8, dtype=np.int64)
    nb_s1 = np.empty(8, dtype=np.int64)
    nb_s2 = np.empty(8, dtype=np.int64)

    nodes_processed = 0
    goal_found = False
    gs_gx = start_gx
    gs_gy = start_gy
    gs_l = start_level
    gs_h = 0

    while hs > 0 and nodes_processed < max_nodes:
        cur_cost = hcost[0]
        cur_ni = np.int64(hstate[0])
        hs -= 1
        hp[0] = hp[hs]
        hcost[0] = hcost[hs]
        hstate[0] = hstate[hs]
        i = 0
        while True:
            l = 2 * i + 1
            r = 2 * i + 2
            sm = i
            if l < hs and hp[l] < hp[sm]:
                sm = l
            if r < hs and hp[r] < hp[sm]:
                sm = r
            if sm == i:
                break
            hp[i], hp[sm] = hp[sm], hp[i]
            hcost[i], hcost[sm] = hcost[sm], hcost[i]
            hstate[i], hstate[sm] = hstate[sm], hstate[i]
            i = sm

        hour_idx = int(cur_ni % max_hours)
        rest = cur_ni // max_hours
        level = int(rest % num_levels)
        rest //= num_levels
        gx = int(rest % grid_w)
        gy = int(rest // grid_w)

        if visited[cur_ni]:
            continue
        visited[cur_ni] = 1
        nodes_processed += 1

        real_x = np.float64(gx * step_size)
        real_y = np.float64(gy * step_size)

        if (abs(real_x - goal_x) <= goal_tolerance and
                abs(real_y - goal_y) <= goal_tolerance and level == goal_level):
            goal_found = True
            gs_gx = gx
            gs_gy = gy
            gs_l = level
            gs_h = hour_idx
            break

        current_total_distance = aux_tdist[cur_ni]
        current_stay = aux_stay[cur_ni]
        current_hour_stay = aux_hstay[cur_ni]

        nb_cnt = 0
        for d in range(8):
            ngx = gx + dir_dx[d]
            ngy = gy + dir_dy[d]
            if ngx < 0 or ngy < 0 or ngx >= grid_w or ngy >= grid_h:
                continue
            nx_real = ngx * step_size
            ny_real = ngy * step_size
            if nx_real < 0 or ny_real < 0 or nx_real >= width or ny_real >= height:
                continue
            nb_dir[nb_cnt] = d
            nb_block[nb_cnt] = (ngy * grid_w + ngx) * block
            if d >= 4:
                nb_s1[nb_cnt] = (gy * grid_w + ngx) * block
                nb_s2[nb_cnt] = (ngy * grid_w + gx) * block
            else:
                nb_s1[nb_cnt] = 0
                nb_s2[nb_cnt] = 0
            nb_cnt += 1

        for k in range(nb_cnt):
            d = nb_dir[k]
            ngx = gx + dir_dx[d]
            ngy = gy + dir_dy[d]

            step_distance = dir_mul[d] * step_size
            length_cost = step_distance * length_penalty
            h_val = hgrid[ngy * grid_w + ngx] * wastar
            nb_ok = has_nb[ngy * grid_w + ngx] != 0

            nblock = nb_block[k]
            diag = d >= 4
            s1b = nb_s1[k]
            s2b = nb_s2[k]

            lvl_lo = -lookahead_levels
            if level + lvl_lo < 0:
                lvl_lo = -level
            lvl_hi = 0 if current_stay > 0.0 else lookahead_levels
            if level + lvl_hi >= num_levels:
                lvl_hi = num_levels - 1 - level

            for lvl_shift in range(lvl_lo, lvl_hi + 1):
                next_level = level + lvl_shift

                if lvl_shift > 0:
                    new_stay = level_stay_multiplier * lvl_shift
                else:
                    new_stay = current_stay - step_distance
                    if new_stay < 0.0:
                        new_stay = 0.0

                level_change_cost = abs(lvl_shift) * LEVEL_CHANGE_PENALTY
                lvl_off = next_level * max_hours

                new_total_distance = current_total_distance + step_distance

                base_hour = int(new_total_distance / flight_speed)
                if base_hour < 0:
                    base_hour = 0
                if base_hour >= max_hours:
                    base_hour = max_hours - 1

                if current_hour_stay > 0.0:
                    hs_begin = 0
                    hs_end = 0
                else:
                    hs_begin = hour_idx - base_hour if hour_idx > base_hour else 0
                    hs_end = hour_lookahead

                for hour_shift in range(hs_begin, hs_end + 1):
                    target_hour = base_hour + hour_shift
                    if target_hour >= max_hours:
                        target_hour = max_hours - 1

                    if target_hour < hour_idx:
                        continue

                    intentional_shift = target_hour - base_hour

                    if intentional_shift > 0:
                        new_hour_stay = hour_stay_distance * intentional_shift
                    else:
                        new_hour_stay = current_hour_stay - step_distance
                        if new_hour_stay < 0.0:
                            new_hour_stay = 0.0

                    total_danger = dgf[nblock + lvl_off + target_hour]
                    if diag:
                        corner = min(dgf[s1b + lvl_off + target_hour],
                                     dgf[s2b + lvl_off + target_hour])
                        if corner > total_danger:
                            total_danger = corner

                    danger_cost = total_danger * danger_weight
                    hour_change_cost = hour_switch_penalty * (
                        intentional_shift if intentional_shift > 0 else 0)

                    added_cost = 0.0
                    stay_out = new_stay
                    loitered = False

                    if intentional_shift > 0:
                        required_distance = np.float64(target_hour) * flight_speed
                        need_extra = required_distance - new_total_distance
                        if need_extra > 0.0 and nb_ok:
                            loitered = True
                            add_c = 0.0
                            dpos = new_total_distance
                            rem = need_extra
                            while rem > 1e-9:
                                hh = int(dpos / flight_speed)
                                if hh > max_hours - 1:
                                    hh = max_hours - 1
                                if hh >= max_hours - 1:
                                    chunk = rem
                                else:
                                    chunk = min(rem, np.float64(hh + 1) * flight_speed - dpos)
                                if chunk <= 0.0:
                                    chunk = rem
                                avg_h = 0.5 * (dgf[nblock + lvl_off + hh] +
                                               mnf[nblock + lvl_off + hh])
                                add_c += chunk * (avg_h * danger_weight / step_distance
                                                  + length_penalty)
                                dpos += chunk
                                rem -= chunk
                            added_cost = add_c
                            new_total_distance += need_extra
                            stay_out = new_stay - need_extra
                            if stay_out < 0.0:
                                stay_out = 0.0

                    step_cost = danger_cost + length_cost + level_change_cost + \
                        hour_change_cost + added_cost
                    new_cost = cur_cost + step_cost

                    ni = nblock + lvl_off + target_hour
                    if visited[ni]:
                        if (not loitered and target_hour == max_hours - 1 and
                                base_hour + hour_shift >= max_hours - 1):
                            break
                        continue
                    if new_cost < dist_grid[ni]:
                        dist_grid[ni] = new_cost
                        aux_stay[ni] = stay_out
                        aux_hstay[ni] = new_hour_stay
                        aux_tdist[ni] = new_total_distance
                        prev_state[ni] = np.uint32(cur_ni)

                        if hs == cap:
                            cap *= 2
                            nhp = np.empty(cap, dtype=np.float64)
                            nhc = np.empty(cap, dtype=np.float64)
                            nhs = np.empty(cap, dtype=np.uint32)
                            nhp[:hs] = hp[:hs]
                            nhc[:hs] = hcost[:hs]
                            nhs[:hs] = hstate[:hs]
                            hp = nhp
                            hcost = nhc
                            hstate = nhs
                        i = hs
                        hp[i] = new_cost + h_val
                        hcost[i] = new_cost
                        hstate[i] = np.uint32(ni)
                        while i > 0:
                            p = (i - 1) // 2
                            if hp[p] <= hp[i]:
                                break
                            hp[p], hp[i] = hp[i], hp[p]
                            hcost[p], hcost[i] = hcost[i], hcost[p]
                            hstate[p], hstate[i] = hstate[i], hstate[p]
                            i = p
                        hs += 1
                    else:
                        if (new_cost == dist_grid[ni] and
                                new_total_distance < aux_tdist[ni]):
                            aux_stay[ni] = stay_out
                            aux_hstay[ni] = new_hour_stay
                            aux_tdist[ni] = new_total_distance
                            prev_state[ni] = np.uint32(cur_ni)

                    if (not loitered and target_hour == max_hours - 1 and
                            base_hour + hour_shift >= max_hours - 1):
                        break

    truncated = nodes_processed >= max_nodes

    if not goal_found:
        best_cost = inf
        for gy2 in range(grid_h):
            for gx2 in range(grid_w):
                lvl2 = goal_level
                for hr2 in range(max_hours):
                    ni2 = ((gy2 * grid_w + gx2) * num_levels + lvl2) * max_hours + hr2
                    if visited[ni2] == 0:
                        continue
                    rx2 = np.float64(gx2 * step_size)
                    ry2 = np.float64(gy2 * step_size)
                    if (abs(rx2 - goal_x) <= goal_tolerance * 3 and
                            abs(ry2 - goal_y) <= goal_tolerance * 3):
                        c2 = dist_grid[ni2] + hgrid[gy2 * grid_w + gx2]
                        if c2 < best_cost:
                            best_cost = c2
                            gs_gx = gx2
                            gs_gy = gy2
                            gs_l = lvl2
                            gs_h = hr2
        if best_cost < inf:
            goal_found = True

    if not goal_found:
        empty = np.zeros(0, dtype=np.float64)
        return (empty, empty, np.zeros(0, dtype=np.int64), empty, empty,
                np.zeros(0, dtype=np.int64), False, truncated)

    cnt = 0
    cur = ((gs_gy * grid_w + gs_gx) * num_levels + gs_l) * max_hours + gs_h
    c = cur
    while True:
        cnt += 1
        pr = prev_state[c]
        if pr == NO_PREV:
            break
        c = np.int64(pr)

    ppx = np.empty(cnt, dtype=np.float64)
    ppy = np.empty(cnt, dtype=np.float64)
    plvl = np.empty(cnt, dtype=np.int64)
    pstay = np.empty(cnt, dtype=np.float64)
    pdist = np.empty(cnt, dtype=np.float64)
    parr = np.empty(cnt, dtype=np.int64)

    c = cur
    pos = cnt - 1
    while True:
        hr = int(c % max_hours)
        lv = int((c // max_hours) % num_levels)
        cc = c // (max_hours * num_levels)
        gxx = int(cc % grid_w)
        gyy = int(cc // grid_w)
        ppx[pos] = np.float64(gxx * step_size)
        ppy[pos] = np.float64(gyy * step_size)
        plvl[pos] = lv
        pstay[pos] = aux_stay[c]
        pdist[pos] = aux_tdist[c]
        parr[pos] = hr
        pos -= 1
        pr = prev_state[c]
        if pr == NO_PREV:
            break
        c = np.int64(pr)

    return ppx, ppy, plvl, pstay, pdist, parr, True, truncated


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


def materialize_loiter_loops(px, py, lvl, stay, tdist, arr, dg, grid_w, grid_h,
                             width, height, step_size, num_levels, max_hours, flight_speed):
    n = len(px)
    frozen = np.zeros(n, dtype=np.uint8)
    if n < 2:
        return px, py, lvl, stay, tdist, arr, frozen
    any_wait = False
    for i in range(n - 1):
        geo = float(np.sqrt((px[i + 1] - px[i]) ** 2 + (py[i + 1] - py[i]) ** 2))
        if tdist[i + 1] - tdist[i] - geo > 1.0:
            any_wait = True
            break
    if not any_wait:
        return px, py, lvl, stay, tdist, arr, frozen

    dx8 = [0, 1, 0, -1, 1, 1, -1, -1]
    dy8 = [1, 0, -1, 0, 1, -1, 1, -1]
    opx = [px[0]]
    opy = [py[0]]
    olv = [int(lvl[0])]
    ofz = [0]

    for i in range(n - 1):
        ax, ay = px[i], py[i]
        bx, by = px[i + 1], py[i + 1]
        geo = float(np.sqrt((bx - ax) ** 2 + (by - ay) ** 2))
        added = (tdist[i + 1] - tdist[i]) - geo
        L = int(lvl[i + 1])

        if added <= 1.0:
            opx.append(bx); opy.append(by); olv.append(L); ofz.append(0)
            continue

        opx.append(bx); opy.append(by); olv.append(L); ofz.append(1)
        bgx = int(np.rint(bx / float(step_size)))
        bgy = int(np.rint(by / float(step_size)))

        dpos = tdist[i] + geo
        rem = added
        while rem > 1.0:
            h = int(dpos / flight_speed)
            if h > max_hours - 1:
                h = max_hours - 1
            if h >= max_hours - 1:
                chunk = rem
            else:
                chunk = min(rem, float(h + 1) * flight_speed - dpos)
            if chunk <= 0.0:
                chunk = rem

            best = 1e18
            nxr, nyr = ax, ay
            for d in range(8):
                gx2 = bgx + dx8[d]
                gy2 = bgy + dy8[d]
                if gx2 < 0 or gy2 < 0 or gx2 >= grid_w or gy2 >= grid_h:
                    continue
                rx = gx2 * step_size
                ry = gy2 * step_size
                if rx >= width or ry >= height:
                    continue
                dv = dg[gy2, gx2, L, h]
                if dv < best:
                    best = dv
                    nxr, nyr = float(rx), float(ry)

            loop_len = 2.0 * float(np.sqrt((nxr - bx) ** 2 + (nyr - by) ** 2))
            loops = int(np.ceil(chunk / loop_len))
            if loops < 1:
                loops = 1
            for _ in range(loops):
                opx.append(nxr); opy.append(nyr); olv.append(L); ofz.append(1)
                opx.append(bx); opy.append(by); olv.append(L); ofz.append(1)
            flown = float(loops) * loop_len
            dpos += flown
            rem -= flown

    m = len(opx)
    otd = np.zeros(m, dtype=np.float64)
    ost = np.zeros(m, dtype=np.float64)
    oar = np.zeros(m, dtype=np.int64)
    for i in range(1, m):
        otd[i] = otd[i - 1] + float(np.sqrt((opx[i] - opx[i - 1]) ** 2 +
                                            (opy[i] - opy[i - 1]) ** 2))
        h = int(otd[i] / flight_speed)
        oar[i] = max_hours - 1 if h > max_hours - 1 else h
        ost[i] = (LEVEL_STAY_MULTIPLIER * (olv[i] - olv[i - 1])
                  if olv[i] > olv[i - 1] else 0.0)
    return (np.array(opx), np.array(opy), np.array(olv, dtype=np.int64),
            ost, otd, oar, np.array(ofz, dtype=np.uint8))


def refine_path(px, py, lvl, stay, tdist, arr, frozen, arrays_count,
                goal_x, goal_y, goal_level, max_step=float(DIJKSTRA_STEP_SIZE)):
    rx, ry, rl, rs, rt, ra, rf = [], [], [], [], [], [], []
    if len(px) == 0:
        return rx, ry, rl, rs, rt, ra, rf

    def hour_of(dist):
        return int(dist_to_hour(dist, arrays_count))

    rx.append(px[0]); ry.append(py[0]); rl.append(int(lvl[0]))
    rs.append(stay[0]); rt.append(tdist[0]); ra.append(int(arr[0])); rf.append(int(frozen[0]))

    for i in range(len(px) - 1):
        sx, sy = px[i], py[i]
        ex, ey = px[i + 1], py[i + 1]
        start_lvl, end_lvl = int(lvl[i]), int(lvl[i + 1])
        start_stay, end_stay = stay[i], stay[i + 1]
        start_dist, end_dist = tdist[i], tdist[i + 1]
        segment_length = float(np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2))

        if segment_length <= max_step:
            rx.append(ex); ry.append(ey); rl.append(end_lvl)
            rs.append(end_stay); rt.append(end_dist); ra.append(int(arr[i + 1]))
            rf.append(int(frozen[i + 1]))
        else:
            num_steps = max(2, int(segment_length / max_step) + 1)
            for step in range(1, num_steps):
                ratio = float(step) / num_steps
                x = sx + ratio * (ex - sx)
                y = sy + ratio * (ey - sy)
                interpolated_dist = start_dist + ratio * (end_dist - start_dist)
                if start_lvl == end_lvl:
                    interpolated_lvl = start_lvl
                    interpolated_stay = max(0.0, start_stay - ratio * segment_length)
                else:
                    if ratio < 0.5:
                        interpolated_lvl = start_lvl
                        interpolated_stay = max(0.0, start_stay - ratio * segment_length)
                    else:
                        interpolated_lvl = end_lvl
                        level_diff = abs(end_lvl - start_lvl)
                        interpolated_stay = max(
                            0.0, LEVEL_STAY_MULTIPLIER * level_diff -
                            (ratio - 0.5) * segment_length)
                rx.append(x); ry.append(y); rl.append(interpolated_lvl)
                rs.append(interpolated_stay); rt.append(interpolated_dist)
                ra.append(hour_of(interpolated_dist))
                rf.append(1 if (frozen[i] and frozen[i + 1]) else 0)
            rx.append(ex); ry.append(ey); rl.append(end_lvl)
            rs.append(end_stay); rt.append(end_dist); ra.append(int(arr[i + 1]))
            rf.append(int(frozen[i + 1]))

    last_x, last_y = rx[-1], ry[-1]
    dist_to_goal = float(np.sqrt((goal_x - last_x) ** 2 + (goal_y - last_y) ** 2))

    if dist_to_goal > max_step:
        num_steps = max(2, int(dist_to_goal / max_step) + 1)
        for step in range(1, num_steps):
            ratio = float(step) / num_steps
            x = last_x + ratio * (goal_x - last_x)
            y = last_y + ratio * (goal_y - last_y)
            interpolated_dist = rt[-1] + ratio * dist_to_goal
            interpolated_lvl = rl[-1]
            interpolated_stay = max(0.0, rs[-1] - ratio * dist_to_goal)
            rx.append(x); ry.append(y); rl.append(interpolated_lvl)
            rs.append(interpolated_stay); rt.append(interpolated_dist)
            ra.append(hour_of(interpolated_dist)); rf.append(0)

    rf.append(0)
    rx.append(goal_x); ry.append(goal_y); rl.append(int(goal_level))
    goal_dist = rt[-1] + float(np.sqrt((goal_x - rx[-2]) ** 2 + (goal_y - ry[-2]) ** 2))
    rt.append(goal_dist)
    rs.append(0.0)
    ra.append(hour_of(goal_dist))

    for i in range(1, len(rs)):
        rs[i] = (LEVEL_STAY_MULTIPLIER * (rl[i] - rl[i - 1])
                 if rl[i] > rl[i - 1] else 0.0)
    if rs:
        rs[0] = 0.0

    return rx, ry, rl, rs, rt, ra, rf


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


def calculate(danger_map, forecasts, points, wastar=1.0, threads=0, seed=RNG_SEED):
    sizes = {len(p) for p in points}
    if sizes == {3}:
        route_levels = [int(p[2]) for p in points]
    elif sizes == {2}:
        route_levels = None
    else:
        raise ValueError("все точки должны быть (y, x) или (y, x, эшелон)")
    if route_levels is not None:
        for lv in route_levels:
            if lv < 0 or lv >= NUM_LEVELS:
                raise RuntimeError("ValueError: level out of range 0..31")

    if threads > 0:
        set_num_threads(threads)

    danger2d = np.ascontiguousarray(danger_map, dtype=np.float64).copy()
    apply_hyperbolic(danger2d)
    fc4 = np.ascontiguousarray(forecasts, dtype=np.float64).copy()
    apply_hyperbolic(fc4)

    height, width = danger2d.shape
    n_hours, f_levels, f_h, f_w = fc4.shape
    scale_y = float(height) / float(f_h)
    scale_x = float(width) / float(f_w)

    route_xy = [(float(p[1]), float(p[0])) for p in points]
    if len(route_xy) < 2:
        return None

    for idx0, (x, y) in enumerate(route_xy):
        if not (0 <= x < width and 0 <= y < height):
            raise RuntimeError(
                "ValueError: Route point %d (%.1f, %.1f) is out of bounds for map %dx%d"
                % (idx0 + 1, x, y, width, height))

    mt = np.zeros(624, dtype=np.uint32)
    ridx = np.zeros(1, dtype=np.int64)
    gauss = np.zeros(2, dtype=np.float64)
    rng_seed(mt, ridx, gauss, seed)

    dk_dang = Dict.empty(types.uint64, types.float64)
    dk_hour = Dict.empty(types.uint64, types.int64)
    order = np.zeros(1200000, dtype=np.uint64)
    ostate = np.zeros(2, dtype=np.int64)

    step = DIJKSTRA_STEP_SIZE
    grid_w = int(np.ceil(float(width) / step)) + 1
    grid_h = int(np.ceil(float(height) / step)) + 1
    fs_scaled = FLIGHT_SPEED * HOURS_PER_ARRAY_SLICE
    hour_lookahead = n_hours - 1 if n_hours > 0 else 0

    dg = build_danger_grid(danger2d, fc4, LEVEL_PENALTIES, grid_w, grid_h, step,
                           NUM_LEVELS, n_hours, width, height, scale_x, scale_y)
    mn, has_nb = build_min_nb(dg, grid_w, grid_h, step, width, height,
                              NUM_LEVELS, n_hours)
    md = build_min_dh(dg, grid_w, grid_h, NUM_LEVELS, n_hours)

    w_eff = wastar if wastar >= 1.0 else 1.0

    seg_results = []
    carry_level = -1
    for si in range(len(route_xy) - 1):
        sx, sy = route_xy[si]
        ex, ey = route_xy[si + 1]

        if route_levels is not None:
            start_level = route_levels[si]
            goal_level = route_levels[si + 1]
        else:
            start_level = carry_level
            goal_level = -1
        if start_level < 0:
            start_level = find_best_level(sx, sy, 0.0, danger2d, fc4, scale_x, scale_y,
                                          dk_dang, dk_hour, order, ostate)
        if goal_level < 0:
            straight = float(np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2))
            goal_level = find_best_level(ex, ey, straight, danger2d, fc4,
                                         scale_x, scale_y, dk_dang, dk_hour,
                                         order, ostate)

        hg = build_hgrid(md, grid_w, grid_h, step, width, height,
                         ex, ey, DIJKSTRA_GOAL_TOLERANCE,
                         SAFETY_WEIGHT * DIJKSTRA_DANGER_WEIGHT,
                         LENGTH_WEIGHT * DIJKSTRA_LENGTH_PENALTY)

        (kx, ky, klv, kst, ktd, kar, ok, truncated) = kernel(
            sx, sy, ex, ey, start_level, goal_level, danger2d, dg, mn, has_nb, hg,
            grid_w, grid_h, width, height, step, LOOKAHEAD_LEVELS,
            LEVEL_STAY_MULTIPLIER, NUM_LEVELS, fs_scaled, n_hours,
            SAFETY_WEIGHT * DIJKSTRA_DANGER_WEIGHT,
            LENGTH_WEIGHT * DIJKSTRA_LENGTH_PENALTY,
            DIJKSTRA_GOAL_TOLERANCE, MAX_NODES, hour_lookahead,
            HOUR_STAY_DISTANCE, HOUR_SWITCH_PENALTY, w_eff)

        if truncated:
            print("Поиск оборван по лимиту узлов, маршрут может быть неоптимален",
                  file=sys.stderr)
        if not ok or kx.shape[0] == 0:
            return None

        mx, my, mlv, mst, mtd, mar, mfz = materialize_loiter_loops(
            list(kx), list(ky), list(klv), list(kst), list(ktd), list(kar),
            dg, grid_w, grid_h, width, height, step, NUM_LEVELS, n_hours, fs_scaled)

        rx, ry, rl, rs, rt, ra, rf = refine_path(
            mx, my, mlv, mst, mtd, mar, mfz, n_hours, ex, ey, goal_level)

        d_length = 0.0
        for i in range(len(rx) - 1):
            d_length += float(np.sqrt((rx[i + 1] - rx[i]) ** 2 + (ry[i + 1] - ry[i]) ** 2))

        safety = SAFETY_WEIGHT * (1.0 + SAFETY_WEIGHT_COEFF * (d_length / BASE_PATH_LENGTH))
        safety = max(SAFETY_WEIGHT, min(safety, SAFETY_WEIGHT * 3.0))

        oxs, oys, olv, otd = optimize_stage2(
            rx, ry, rl, rs, rt, ra, rf, safety, LENGTH_WEIGHT,
            danger2d, fc4, scale_x, scale_y, mt, ridx, gauss,
            dk_dang, dk_hour, order, ostate)

        seg_results.append((list(rx), list(ry), [int(v) for v in rl],
                            list(oxs), list(oys), [int(v) for v in olv],
                            goal_level))
        carry_level = goal_level

    c_dx, c_dy, c_dl = [], [], []
    c_ox, c_oy, c_ol = [], [], []
    for idx0, seg in enumerate(seg_results):
        start = 1 if idx0 > 0 else 0
        c_dx += seg[0][start:]
        c_dy += seg[1][start:]
        c_dl += seg[2][start:]
        c_ox += seg[3][start:]
        c_oy += seg[4][start:]
        c_ol += seg[5][start:]

    px_, py_, lv_ = (c_ox, c_oy, c_ol) if c_ox else (c_dx, c_dy, c_dl)
    if not px_:
        return None
    if not lv_:
        lv_ = c_dl

    out = np.empty((len(px_), 3), dtype=np.int64)
    for i in range(len(px_)):
        out[i, 0] = np.int64(np.rint(py_[i]))
        out[i, 1] = np.int64(np.rint(px_[i]))
        out[i, 2] = np.int64(lv_[i]) if i < len(lv_) else 0
    return out


def main():
    args = sys.argv[1:]
    wastar, threads, seed = 1.0, 0, RNG_SEED
    levels = None
    plain = []
    for a in args:
        if a.startswith("--wastar="):
            wastar = float(a[9:])
        elif a.startswith("--threads="):
            threads = int(a[10:])
        elif a.startswith("--seed="):
            seed = int(a[7:])
        elif a.startswith("--levels="):
            levels = [int(x) for x in a[9:].split(",")]
        else:
            plain.append(float(a))
    if len(plain) >= 4 and len(plain) % 2 == 0:
        points = [(plain[i], plain[i + 1]) for i in range(0, len(plain), 2)]
    else:
        points = [(2000.0, 8000.0), (3000.0, 3850.0), (1500.0, 1500.0)]
    if levels is not None:
        if len(levels) != len(points):
            raise ValueError("--levels: число эшелонов должно совпадать с числом точек")
        points = [(y, x, lv) for (y, x), lv in zip(points, levels)]

    from pathlib import Path
    data = Path("data")
    danger_map = np.load(data / "map-test.npy").astype(np.float64)
    forecasts = np.stack([np.load(data / f"{h}h.npy").astype(np.float64)
                          for h in range(10)])

    t0 = time.perf_counter()
    route = calculate(danger_map, forecasts, points, wastar, threads, seed)
    dt = time.perf_counter() - t0

    print(f"Время выполнения: {dt:.6f} c")
    if route is None:
        print("Маршрут не построен", file=sys.stderr)
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(danger_map, cmap="gray_r", origin="upper", alpha=0.8)
        ax.plot(route[:, 1], route[:, 0], "r-", linewidth=1.2, alpha=0.9)
        ax.scatter(points[0][1], points[0][0], color="green", s=200, marker="*")
        ax.scatter(points[-1][1], points[-1][0], color="blue", s=200, marker="*")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig("cpp_test.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
