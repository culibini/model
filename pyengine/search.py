import numpy as np
from numba import njit

from .config import LEVEL_CHANGE_PENALTY


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
