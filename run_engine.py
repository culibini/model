import ctypes
import sys
import time
from pathlib import Path

import numpy as np


def load_library():
    here = Path(__file__).resolve().parent
    return ctypes.CDLL(str(here / "engine" / "libengine.so"))


def bind(lib):
    lib.engine_calculate_path.restype = ctypes.POINTER(ctypes.c_longlong)
    lib.engine_calculate_path.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.c_uint, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.engine_free.restype = None
    lib.engine_free.argtypes = [ctypes.POINTER(ctypes.c_longlong)]


def parse_args(argv):
    points = []
    wastar, threads, seed = 1.0, 0, 12345
    plain = []
    for a in argv:
        if a.startswith("--wastar="):
            wastar = float(a[9:])
        elif a.startswith("--threads="):
            threads = int(a[10:])
        elif a.startswith("--seed="):
            seed = int(a[7:])
        else:
            plain.append(float(a))
    if len(plain) >= 4 and len(plain) % 2 == 0:
        points = [(plain[i], plain[i + 1]) for i in range(0, len(plain), 2)]
    else:
        points = [(2000.0, 8000.0), (3000.0, 3850.0), (1500.0, 1500.0)]
    return points, wastar, threads, seed


def calculate(lib, danger_map, forecasts, points, wastar, threads, seed):
    m = np.ascontiguousarray(danger_map, dtype=np.float64)
    f = np.ascontiguousarray(forecasts, dtype=np.float64)
    p = np.ascontiguousarray(np.array(points, dtype=np.float64).reshape(-1))
    out_n = ctypes.c_int(0)
    t0 = time.perf_counter()
    ptr = lib.engine_calculate_path(
        m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m.shape[0], m.shape[1],
        f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), f.shape[0],
        f.shape[1], f.shape[2], f.shape[3],
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(points),
        wastar, threads, seed, ctypes.byref(out_n),
    )
    dt = time.perf_counter() - t0
    n = out_n.value
    if n < 0:
        raise RuntimeError("ошибка построения маршрута (проверьте точки и данные)")
    if n == 0:
        return None, dt
    route = np.ctypeslib.as_array(ptr, shape=(n, 3)).copy()
    lib.engine_free(ptr)
    return route, dt


def visualize(danger_map, route, points, out_path):
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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    points, wastar, threads, seed = parse_args(sys.argv[1:])
    data = Path("data")
    danger_map = np.load(data / "map-test.npy").astype(np.float64)
    forecasts = np.stack([
        np.load(data / f"{h}h.npy").astype(np.float64) for h in range(10)
    ])

    lib = load_library()
    bind(lib)
    route, dt = calculate(lib, danger_map, forecasts, points, wastar, threads, seed)

    print(f"Время выполнения: {dt:.6f} c")
    if route is None:
        print("Маршрут не построен", file=sys.stderr)
        sys.exit(1)

    visualize(danger_map, route, points, "cpp_test.png")


if __name__ == "__main__":
    main()
