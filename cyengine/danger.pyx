# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
import numpy as np

from cython.parallel cimport prange
from libc.math cimport INFINITY, rint
from libc.stdint cimport int64_t, uint64_t
from libcpp.unordered_map cimport unordered_map

from . import config as _cfg

cdef double DANGER_SOFT_LIMIT = _cfg.DANGER_SOFT_LIMIT
cdef double DANGER_HYPERBOLIC_RANGE = _cfg.DANGER_HYPERBOLIC_RANGE
cdef double DANGER_HYPERBOLIC_SCALE = _cfg.DANGER_HYPERBOLIC_SCALE
cdef double FLIGHT_SPEED = _cfg.FLIGHT_SPEED
cdef double HOURS_PER_ARRAY_SLICE = _cfg.HOURS_PER_ARRAY_SLICE
cdef double LEVEL_STAY_MULTIPLIER = _cfg.LEVEL_STAY_MULTIPLIER
cdef long long NUM_LEVELS = _cfg.NUM_LEVELS
cdef long long LEVEL_OPTIMIZATION_RANGE = _cfg.LEVEL_OPTIMIZATION_RANGE

_LEVEL_PENALTIES = np.ascontiguousarray(_cfg.LEVEL_PENALTIES, dtype=np.float64)
cdef double[::1] LEVEL_PENALTIES = _LEVEL_PENALTIES


cdef double hyperbolic_danger_c(double v) noexcept nogil:
    cdef double delta, cap, capped, denom
    if DANGER_HYPERBOLIC_RANGE <= 0.0:
        return v
    delta = v - DANGER_SOFT_LIMIT
    if delta <= 0.0:
        return v
    cap = DANGER_HYPERBOLIC_RANGE * 0.999999
    capped = delta if delta < cap else cap
    denom = DANGER_HYPERBOLIC_RANGE - capped
    if denom < 1e-6:
        denom = 1e-6
    return v + DANGER_HYPERBOLIC_SCALE * (capped / denom)


def hyperbolic_danger(double v):
    return hyperbolic_danger_c(v)


def apply_hyperbolic(a):
    flat = a.reshape(-1)
    cdef double[::1] mv = flat
    cdef Py_ssize_t i, n = mv.shape[0]
    for i in prange(n, nogil=True):
        mv[i] = hyperbolic_danger_c(mv[i])


cdef double sample3d(const double[:, :, :, ::1] fc4, long long hour, long long level,
                     double px, double py, double scale_x, double scale_y) noexcept nogil:
    cdef long long L = fc4.shape[1]
    cdef long long h3 = fc4.shape[2]
    cdef long long w3 = fc4.shape[3]
    cdef double u, v, mu, mv, fu, fv, d00, d10, d01, d11
    cdef long long x0, y0, x1, y1
    if level < 0 or level >= L:
        return 0.0
    u = px / scale_x - 0.5
    v = py / scale_y - 0.5
    if u < 0.0:
        u = 0.0
    if v < 0.0:
        v = 0.0
    mu = <double>w3 - 1.0
    mv = <double>h3 - 1.0
    if u > mu:
        u = mu
    if v > mv:
        v = mv
    x0 = <long long>u
    y0 = <long long>v
    x1 = x0 + 1 if x0 + 1 < w3 - 1 else w3 - 1
    y1 = y0 + 1 if y0 + 1 < h3 - 1 else h3 - 1
    fu = u - x0
    fv = v - y0
    d00 = fc4[hour, level, y0, x0]
    d10 = fc4[hour, level, y0, x1]
    d01 = fc4[hour, level, y1, x0]
    d11 = fc4[hour, level, y1, x1]
    return (d00 * (1.0 - fu) + d10 * fu) * (1.0 - fv) + (d01 * (1.0 - fu) + d11 * fu) * fv


def sample3d_py(fc4, long long hour, long long level, double px, double py,
                double scale_x, double scale_y):
    return sample3d(fc4, hour, level, px, py, scale_x, scale_y)


cpdef long long dist_to_hour(double dist, long long arrays_count):
    cdef double time_hours = dist / FLIGHT_SPEED
    cdef long long hour_idx = <long long>(time_hours / HOURS_PER_ARRAY_SLICE)
    cdef long long max_idx
    if hour_idx < 0:
        return 0
    max_idx = arrays_count - 1
    return hour_idx if hour_idx <= max_idx else max_idx


cdef class DangerCache:
    cdef unordered_map[uint64_t, double] dang
    cdef unordered_map[uint64_t, int64_t] hour
    cdef uint64_t[::1] order
    cdef Py_ssize_t o0
    cdef Py_ssize_t o1
    cdef object _order_ref
    cdef object _d2_ref
    cdef object _fc4_ref
    cdef double[:, ::1] danger2d
    cdef const double[:, :, :, ::1] fc4
    cdef double scale_x
    cdef double scale_y

    def __init__(self, danger2d, fc4, double scale_x, double scale_y):
        self._d2_ref = danger2d
        self._fc4_ref = fc4
        self.danger2d = danger2d
        self.fc4 = fc4
        self.scale_x = scale_x
        self.scale_y = scale_y
        self._order_ref = np.zeros(1200000, dtype=np.uint64)
        self.order = self._order_ref
        self.o0 = 0
        self.o1 = 0

    cdef double _total(self, double px, double py, long long level, double dist,
                       int64_t* hour_out):
        cdef int64_t x = <int64_t>rint(px)
        cdef int64_t y = <int64_t>rint(py)
        cdef long long n_hours = self.fc4.shape[0]
        cdef long long hour_idx = dist_to_hour(dist, n_hours)
        cdef uint64_t key = ((<uint64_t>x << 40) | (<uint64_t>y << 20) |
                             (<uint64_t>level << 8) | <uint64_t>hour_idx)
        cdef long long h2, w2, yi, xi
        cdef double base, penalty, level_danger, total
        cdef Py_ssize_t keep, i, to_drop
        cdef uint64_t k

        if self.dang.count(key):
            hour_out[0] = self.hour[key]
            return self.dang[key]

        h2 = self.danger2d.shape[0]
        w2 = self.danger2d.shape[1]
        yi = <long long>rint(py)
        xi = <long long>rint(px)
        if xi < 0 or xi >= w2 or yi < 0 or yi >= h2:
            base = 1e9
        else:
            base = self.danger2d[yi, xi]
        penalty = LEVEL_PENALTIES[level] if 0 <= level < NUM_LEVELS else 0.0
        level_danger = 0.0
        if 0 <= hour_idx < n_hours:
            level_danger = sample3d(self.fc4, hour_idx, level, px, py,
                                    self.scale_x, self.scale_y)
        total = base + level_danger + penalty
        self.dang[key] = total
        self.hour[key] = <int64_t>hour_idx

        if self.o1 >= self.order.shape[0]:
            keep = self.o1 - self.o0
            for i in range(keep):
                self.order[i] = self.order[self.o0 + i]
            self.o1 = keep
            self.o0 = 0
        self.order[self.o1] = key
        self.o1 += 1

        if <Py_ssize_t>self.dang.size() > 500000:
            to_drop = self.o1 - self.o0
            if to_drop > 50000:
                to_drop = 50000
            for i in range(to_drop):
                k = self.order[self.o0 + i]
                if self.dang.count(k):
                    self.dang.erase(k)
                    self.hour.erase(k)
            self.o0 += to_drop
            if self.o0 > self.o1 // 2:
                keep = self.o1 - self.o0
                for i in range(keep):
                    self.order[i] = self.order[self.o0 + i]
                self.o1 = keep
                self.o0 = 0

        hour_out[0] = <int64_t>hour_idx
        return total

    def total_danger(self, double px, double py, long long level, double dist):
        cdef int64_t h = 0
        cdef double d = self._total(px, py, level, dist, &h)
        return d, h

    cpdef long long find_best_level(self, double px, double py, double dist):
        cdef long long best_level = 0
        cdef double best_danger = INFINITY
        cdef long long level
        cdef int64_t h = 0
        cdef double d
        for level in range(NUM_LEVELS):
            d = self._total(px, py, level, dist, &h)
            if d < best_danger:
                best_danger = d
                best_level = level
        return best_level

    cpdef long long find_optimal_level(self, double px, double py,
                                       long long current_level,
                                       double current_level_distance,
                                       double total_distance):
        cdef long long best_level = current_level
        cdef int64_t bh = 0
        cdef int64_t h = 0
        cdef double bd = self._total(px, py, current_level, total_distance, &bh)
        cdef long long off, cand
        cdef double d
        for off in range(-LEVEL_OPTIMIZATION_RANGE, LEVEL_OPTIMIZATION_RANGE + 1):
            cand = current_level + off
            if cand < 0 or cand >= NUM_LEVELS:
                continue
            if cand > current_level:
                if current_level_distance < LEVEL_STAY_MULTIPLIER * (cand - current_level):
                    continue
            d = self._total(px, py, cand, total_distance, &h)
            if d < bd or (d == bd and h < bh):
                bd = d
                bh = h
                best_level = cand
        return best_level
