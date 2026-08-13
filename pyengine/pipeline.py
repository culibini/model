import sys

import numpy as np
from numba import set_num_threads
from numba import types
from numba.typed import Dict

from .config import (BASE_PATH_LENGTH, DIJKSTRA_DANGER_WEIGHT,
                     DIJKSTRA_GOAL_TOLERANCE, DIJKSTRA_LENGTH_PENALTY,
                     DIJKSTRA_STEP_SIZE, FLIGHT_SPEED, HOURS_PER_ARRAY_SLICE,
                     HOUR_STAY_DISTANCE, HOUR_SWITCH_PENALTY, LENGTH_WEIGHT,
                     LEVEL_PENALTIES, LEVEL_STAY_MULTIPLIER, LOOKAHEAD_LEVELS,
                     MAX_NODES, NUM_LEVELS, RNG_SEED, SAFETY_WEIGHT,
                     SAFETY_WEIGHT_COEFF)
from .danger import apply_hyperbolic, find_best_level
from .geometry import materialize_loiter_loops, refine_path
from .optimize import optimize_stage2
from .precompute import build_danger_grid, build_hgrid, build_min_dh, build_min_nb
from .rng import rng_seed
from .search import kernel


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
