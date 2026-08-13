import numpy as np
from numba import njit, prange

from .danger import sample3d


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
