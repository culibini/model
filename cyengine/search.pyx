# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np

from libc.math cimport fabs, rint, sqrt
from libc.stdint cimport uint32_t
from libc.stdlib cimport free, malloc, realloc

from . import config as _cfg

cdef double LEVEL_CHANGE_PENALTY = _cfg.LEVEL_CHANGE_PENALTY

cdef uint32_t NO_PREV = <uint32_t>0xFFFFFFFF


cdef class SearchHeap:
    cdef double* f
    cdef double* c
    cdef uint32_t* s
    cdef long long n
    cdef long long cap

    def __cinit__(self, long long cap):
        self.f = <double*>malloc(cap * sizeof(double))
        self.c = <double*>malloc(cap * sizeof(double))
        self.s = <uint32_t*>malloc(cap * sizeof(uint32_t))
        self.n = 0
        self.cap = cap

    def __dealloc__(self):
        free(self.f)
        free(self.c)
        free(self.s)

    cdef void push(self, double f, double c, uint32_t s) noexcept:
        cdef long long i, p
        cdef double tf
        cdef uint32_t ts
        if self.n == self.cap:
            self.cap *= 2
            self.f = <double*>realloc(self.f, self.cap * sizeof(double))
            self.c = <double*>realloc(self.c, self.cap * sizeof(double))
            self.s = <uint32_t*>realloc(self.s, self.cap * sizeof(uint32_t))
        i = self.n
        self.f[i] = f
        self.c[i] = c
        self.s[i] = s
        while i > 0:
            p = (i - 1) // 2
            if self.f[p] <= self.f[i]:
                break
            tf = self.f[p]; self.f[p] = self.f[i]; self.f[i] = tf
            tf = self.c[p]; self.c[p] = self.c[i]; self.c[i] = tf
            ts = self.s[p]; self.s[p] = self.s[i]; self.s[i] = ts
            i = p
        self.n += 1

    cdef void remove_top(self) noexcept:
        cdef long long i, l, r, sm
        cdef double tf
        cdef uint32_t ts
        self.n -= 1
        self.f[0] = self.f[self.n]
        self.c[0] = self.c[self.n]
        self.s[0] = self.s[self.n]
        i = 0
        while True:
            l = 2 * i + 1
            r = 2 * i + 2
            sm = i
            if l < self.n and self.f[l] < self.f[sm]:
                sm = l
            if r < self.n and self.f[r] < self.f[sm]:
                sm = r
            if sm == i:
                break
            tf = self.f[i]; self.f[i] = self.f[sm]; self.f[sm] = tf
            tf = self.c[i]; self.c[i] = self.c[sm]; self.c[sm] = tf
            ts = self.s[i]; self.s[i] = self.s[sm]; self.s[sm] = ts
            i = sm


cdef double loiter_added_cost(const double[::1] dgf, const double[::1] mnf,
                              long long nblock, long long lvl_off,
                              double start_distance, double need_extra,
                              long long max_hours, double flight_speed,
                              double danger_weight, double length_penalty,
                              double step_distance) noexcept:
    cdef double add_c = 0.0
    cdef double dpos = start_distance
    cdef double rem = need_extra
    cdef double chunk, avg_h, lim
    cdef long long hh
    while rem > 1e-9:
        hh = <long long>(dpos / flight_speed)
        if hh > max_hours - 1:
            hh = max_hours - 1
        if hh >= max_hours - 1:
            chunk = rem
        else:
            lim = <double>(hh + 1) * flight_speed - dpos
            chunk = rem if rem < lim else lim
        if chunk <= 0.0:
            chunk = rem
        avg_h = 0.5 * (dgf[nblock + lvl_off + hh] + mnf[nblock + lvl_off + hh])
        add_c += chunk * (avg_h * danger_weight / step_distance + length_penalty)
        dpos += chunk
        rem -= chunk
    return add_c


def kernel(double start_x, double start_y, double goal_x, double goal_y,
           long long start_level, long long goal_level,
           danger2d, dg, mn, has_nb_arr, hgrid_arr,
           long long grid_w, long long grid_h, long long width, long long height,
           long long step_size, long long lookahead_levels,
           double level_stay_multiplier, long long num_levels,
           double flight_speed, long long max_hours,
           double danger_weight, double length_penalty,
           double goal_tolerance, long long max_nodes, long long hour_lookahead,
           double hour_stay_distance, double hour_switch_penalty, double wastar):
    cdef long long n4 = grid_h * grid_w * num_levels * max_hours
    cdef const double[::1] dgf = dg.reshape(-1)
    cdef const double[::1] mnf = mn.reshape(-1)
    cdef const unsigned char[::1] has_nb = has_nb_arr
    cdef const double[::1] hgrid = hgrid_arr
    cdef long long block = num_levels * max_hours

    cdef long long start_gx = <long long>rint(start_x / <double>step_size)
    cdef long long start_gy = <long long>rint(start_y / <double>step_size)

    cdef double inf = 1e18
    dist_arr = np.full(n4, inf, dtype=np.float64)
    stay_arr = np.zeros(n4, dtype=np.float64)
    hstay_arr = np.zeros(n4, dtype=np.float64)
    tdist_arr = np.zeros(n4, dtype=np.float64)
    prev_arr = np.full(n4, 0xFFFFFFFF, dtype=np.uint32)
    visited_arr = np.zeros(n4, dtype=np.uint8)
    cdef double[::1] dist_grid = dist_arr
    cdef double[::1] aux_stay = stay_arr
    cdef double[::1] aux_hstay = hstay_arr
    cdef double[::1] aux_tdist = tdist_arr
    cdef unsigned int[::1] prev_state = prev_arr
    cdef unsigned char[::1] visited = visited_arr

    cdef long long cap = (max_nodes if max_nodes < n4 else n4) + 10
    cdef SearchHeap heap = SearchHeap(cap)

    cdef long long start_ni = ((start_gy * grid_w + start_gx) * num_levels +
                               start_level) * max_hours + 0
    dist_grid[start_ni] = 0.0
    heap.push(0.0, 0.0, <uint32_t>start_ni)

    cdef double s2 = sqrt(2.0)
    cdef long long[8] dir_dx = [0, 1, 0, -1, 1, 1, -1, -1]
    cdef long long[8] dir_dy = [1, 0, -1, 0, 1, -1, 1, -1]
    cdef double[8] dir_mul = [1.0, 1.0, 1.0, 1.0, s2, s2, s2, s2]

    cdef long long[8] nb_dir
    cdef long long[8] nb_block
    cdef long long[8] nb_s1
    cdef long long[8] nb_s2

    cdef long long nodes_processed = 0
    cdef bint goal_found = False
    cdef long long gs_gx = start_gx
    cdef long long gs_gy = start_gy
    cdef long long gs_l = start_level
    cdef long long gs_h = 0

    cdef double cur_cost, real_x, real_y
    cdef double current_total_distance, current_stay, current_hour_stay
    cdef long long cur_ni, hour_idx, rest, level, gx, gy
    cdef long long nb_cnt, d, ngx, ngy, nx_real, ny_real, k
    cdef double step_distance, length_cost, h_val
    cdef bint nb_ok, diag, loitered
    cdef long long nblock, s1b, s2b
    cdef long long lvl_lo, lvl_hi, lvl_shift, next_level, lvl_off
    cdef double new_stay, level_change_cost, new_total_distance
    cdef long long base_hour, hs_begin, hs_end, hour_shift, target_hour
    cdef long long intentional_shift
    cdef double new_hour_stay, total_danger, corner, danger_cost, hour_change_cost
    cdef double added_cost, stay_out, required_distance, need_extra
    cdef double step_cost, new_cost
    cdef long long ni
    cdef long long gy2, gx2, lvl2, hr2, ni2
    cdef double best_cost, rx2, ry2, c2

    while heap.n > 0 and nodes_processed < max_nodes:
        cur_cost = heap.c[0]
        cur_ni = <long long>heap.s[0]
        heap.remove_top()

        hour_idx = cur_ni % max_hours
        rest = cur_ni // max_hours
        level = rest % num_levels
        rest //= num_levels
        gx = rest % grid_w
        gy = rest // grid_w

        if visited[cur_ni]:
            continue
        visited[cur_ni] = 1
        nodes_processed += 1

        real_x = <double>(gx * step_size)
        real_y = <double>(gy * step_size)

        if (fabs(real_x - goal_x) <= goal_tolerance and
                fabs(real_y - goal_y) <= goal_tolerance and level == goal_level):
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

                level_change_cost = (lvl_shift if lvl_shift >= 0 else -lvl_shift) * \
                    LEVEL_CHANGE_PENALTY
                lvl_off = next_level * max_hours

                new_total_distance = current_total_distance + step_distance

                base_hour = <long long>(new_total_distance / flight_speed)
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
                        corner = dgf[s1b + lvl_off + target_hour]
                        if dgf[s2b + lvl_off + target_hour] < corner:
                            corner = dgf[s2b + lvl_off + target_hour]
                        if corner > total_danger:
                            total_danger = corner

                    danger_cost = total_danger * danger_weight
                    hour_change_cost = hour_switch_penalty * (
                        intentional_shift if intentional_shift > 0 else 0)

                    added_cost = 0.0
                    stay_out = new_stay
                    loitered = False

                    if intentional_shift > 0:
                        required_distance = <double>target_hour * flight_speed
                        need_extra = required_distance - new_total_distance
                        if need_extra > 0.0 and nb_ok:
                            loitered = True
                            added_cost = loiter_added_cost(
                                dgf, mnf, nblock, lvl_off, new_total_distance,
                                need_extra, max_hours, flight_speed,
                                danger_weight, length_penalty, step_distance)
                            new_total_distance = new_total_distance + need_extra
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
                        prev_state[ni] = <uint32_t>cur_ni
                        heap.push(new_cost + h_val, new_cost, <uint32_t>ni)
                    else:
                        if (new_cost == dist_grid[ni] and
                                new_total_distance < aux_tdist[ni]):
                            aux_stay[ni] = stay_out
                            aux_hstay[ni] = new_hour_stay
                            aux_tdist[ni] = new_total_distance
                            prev_state[ni] = <uint32_t>cur_ni

                    if (not loitered and target_hour == max_hours - 1 and
                            base_hour + hour_shift >= max_hours - 1):
                        break

    cdef bint truncated = nodes_processed >= max_nodes

    if not goal_found:
        best_cost = inf
        for gy2 in range(grid_h):
            for gx2 in range(grid_w):
                lvl2 = goal_level
                for hr2 in range(max_hours):
                    ni2 = ((gy2 * grid_w + gx2) * num_levels + lvl2) * max_hours + hr2
                    if visited[ni2] == 0:
                        continue
                    rx2 = <double>(gx2 * step_size)
                    ry2 = <double>(gy2 * step_size)
                    if (fabs(rx2 - goal_x) <= goal_tolerance * 3 and
                            fabs(ry2 - goal_y) <= goal_tolerance * 3):
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

    cdef long long cnt = 0
    cdef long long cur = ((gs_gy * grid_w + gs_gx) * num_levels + gs_l) * max_hours + gs_h
    cdef long long c = cur
    cdef uint32_t pr
    while True:
        cnt += 1
        pr = prev_state[c]
        if pr == NO_PREV:
            break
        c = <long long>pr

    ppx_arr = np.empty(cnt, dtype=np.float64)
    ppy_arr = np.empty(cnt, dtype=np.float64)
    plvl_arr = np.empty(cnt, dtype=np.int64)
    pstay_arr = np.empty(cnt, dtype=np.float64)
    pdist_arr = np.empty(cnt, dtype=np.float64)
    parr_arr = np.empty(cnt, dtype=np.int64)
    cdef double[::1] ppx = ppx_arr
    cdef double[::1] ppy = ppy_arr
    cdef long long[::1] plvl = plvl_arr
    cdef double[::1] pstay = pstay_arr
    cdef double[::1] pdist = pdist_arr
    cdef long long[::1] parr = parr_arr

    cdef long long pos = cnt - 1
    cdef long long hr, lv, cc, gxx, gyy
    c = cur
    while True:
        hr = c % max_hours
        lv = (c // max_hours) % num_levels
        cc = c // (max_hours * num_levels)
        gxx = cc % grid_w
        gyy = cc // grid_w
        ppx[pos] = <double>(gxx * step_size)
        ppy[pos] = <double>(gyy * step_size)
        plvl[pos] = lv
        pstay[pos] = aux_stay[c]
        pdist[pos] = aux_tdist[c]
        parr[pos] = hr
        pos -= 1
        pr = prev_state[c]
        if pr == NO_PREV:
            break
        c = <long long>pr

    return ppx_arr, ppy_arr, plvl_arr, pstay_arr, pdist_arr, parr_arr, True, truncated
