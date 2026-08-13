# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np

cimport openmp
from cython.parallel cimport prange
from libc.math cimport fabs, sqrt

from .danger cimport sample3d


def set_threads(int n):
    openmp.omp_set_num_threads(n)


def build_danger_grid(double[:, ::1] danger2d, const double[:, :, :, ::1] fc4,
                      double[::1] penalties, long long grid_w, long long grid_h,
                      long long step_size, long long num_levels, long long max_hours,
                      long long width, long long height,
                      double scale_x, double scale_y):
    dg_arr = np.zeros((grid_h, grid_w, num_levels, max_hours), dtype=np.float64)
    cdef double[:, :, :, ::1] dg = dg_arr
    cdef long long gy, gx, nx_real, ny_real, level, hour
    cdef double base, lp, ld
    for gy in prange(grid_h, nogil=True):
        for gx in range(grid_w):
            nx_real = gx * step_size
            ny_real = gy * step_size
            if nx_real >= width or ny_real >= height:
                continue
            base = danger2d[ny_real, nx_real]
            for level in range(num_levels):
                lp = penalties[level]
                for hour in range(max_hours):
                    ld = sample3d(fc4, hour, level, <double>nx_real,
                                  <double>ny_real, scale_x, scale_y)
                    dg[gy, gx, level, hour] = base + ld + lp
    return dg_arr


def build_min_nb(const double[:, :, :, ::1] dg, long long grid_w, long long grid_h,
                 long long step_size, long long width, long long height,
                 long long num_levels, long long max_hours):
    mn_arr = np.zeros((grid_h, grid_w, num_levels, max_hours), dtype=np.float64)
    has_nb_arr = np.zeros(grid_h * grid_w, dtype=np.uint8)
    cdef double[:, :, :, ::1] mn = mn_arr
    cdef unsigned char[::1] has_nb = has_nb_arr
    cdef long long[8] dxs = [0, 1, 0, -1, 1, 1, -1, -1]
    cdef long long[8] dys = [1, 0, -1, 0, 1, -1, 1, -1]
    cdef long long gy, gx, b, bx, by, bxr, byr, l, h
    cdef bint first
    cdef double v
    for gy in prange(grid_h, nogil=True):
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
    return mn_arr, has_nb_arr


def build_min_dh(const double[:, :, :, ::1] dg, long long grid_w, long long grid_h,
                 long long num_levels, long long max_hours):
    md_arr = np.empty(grid_h * grid_w, dtype=np.float64)
    cdef double[::1] md = md_arr
    cdef long long gy, gx, l, h
    cdef double m, v
    for gy in prange(grid_h, nogil=True):
        for gx in range(grid_w):
            m = dg[gy, gx, 0, 0]
            for l in range(num_levels):
                for h in range(max_hours):
                    v = dg[gy, gx, l, h]
                    if v < m:
                        m = v
            md[gy * grid_w + gx] = m
    return md_arr


def build_hgrid(const double[::1] md, long long grid_w, long long grid_h,
                long long step_size, long long width, long long height,
                double goal_x, double goal_y, double goal_tolerance,
                double danger_weight, double length_penalty):
    cdef long long n2 = grid_h * grid_w
    hg_arr = np.full(n2, 1e18, dtype=np.float64)
    cdef double[::1] hg = hg_arr
    cdef long long cap = 4 * n2 + 16
    hp_arr = np.empty(cap, dtype=np.float64)
    hc_arr = np.empty(cap, dtype=np.int64)
    cdef double[::1] hp = hp_arr
    cdef long long[::1] hc = hc_arr
    cdef long long hs = 0
    cdef long long gy, gx, c, i, p, l, r, sm, cgx, cgy, d, agx, agy, arx, ary, ai
    cdef long long top_c
    cdef double rx, ry, s2, top_p, enter_c, mul, cand, tf
    cdef long long tc
    cdef long long[8] dxs = [0, 1, 0, -1, 1, 1, -1, -1]
    cdef long long[8] dys = [1, 0, -1, 0, 1, -1, 1, -1]

    for gy in range(grid_h):
        for gx in range(grid_w):
            rx = <double>(gx * step_size)
            ry = <double>(gy * step_size)
            if rx >= width or ry >= height:
                continue
            if fabs(rx - goal_x) <= goal_tolerance and fabs(ry - goal_y) <= goal_tolerance:
                c = gy * grid_w + gx
                hg[c] = 0.0
                i = hs
                hp[i] = 0.0
                hc[i] = c
                while i > 0:
                    p = (i - 1) // 2
                    if hp[p] <= hp[i]:
                        break
                    tf = hp[p]; hp[p] = hp[i]; hp[i] = tf
                    tc = hc[p]; hc[p] = hc[i]; hc[i] = tc
                    i = p
                hs += 1

    s2 = sqrt(2.0)
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
            tf = hp[i]; hp[i] = hp[sm]; hp[sm] = tf
            tc = hc[i]; hc[i] = hc[sm]; hc[sm] = tc
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
                    tf = hp[p]; hp[p] = hp[i]; hp[i] = tf
                    tc = hc[p]; hc[p] = hc[i]; hc[i] = tc
                    i = p
                hs += 1
    return hg_arr
