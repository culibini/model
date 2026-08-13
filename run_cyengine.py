import sys
import time

import numpy as np

from cyengine import RNG_SEED, calculate


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
