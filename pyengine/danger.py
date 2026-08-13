import numpy as np
from numba import njit, prange

from .config import (DANGER_HYPERBOLIC_RANGE, DANGER_HYPERBOLIC_SCALE,
                     DANGER_SOFT_LIMIT, FLIGHT_SPEED, HOURS_PER_ARRAY_SLICE,
                     LEVEL_OPTIMIZATION_RANGE, LEVEL_PENALTIES,
                     LEVEL_STAY_MULTIPLIER, NUM_LEVELS)


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
