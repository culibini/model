import numpy as np
from numba import njit

from .config import DIJKSTRA_STEP_SIZE, LEVEL_STAY_MULTIPLIER
from .danger import dist_to_hour


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
