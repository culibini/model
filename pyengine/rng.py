import numpy as np
from numba import njit


@njit(cache=True)
def rk_next(mt, idx):
    if idx[0] >= 624:
        for i in range(624):
            y = (mt[i] & np.uint32(0x80000000)) | (mt[(i + 1) % 624] & np.uint32(0x7FFFFFFF))
            v = mt[(i + 397) % 624] ^ (y >> np.uint32(1))
            if y & np.uint32(1):
                v ^= np.uint32(0x9908B0DF)
            mt[i] = v
        idx[0] = 0
    y = mt[idx[0]]
    idx[0] += 1
    y ^= y >> np.uint32(11)
    y ^= (y << np.uint32(7)) & np.uint32(0x9D2C5680)
    y ^= (y << np.uint32(15)) & np.uint32(0xEFC60000)
    y ^= y >> np.uint32(18)
    return y


@njit(cache=True)
def rng_seed(mt, idx, gauss, seed):
    mt[0] = np.uint32(seed)
    for i in range(1, 624):
        mt[i] = np.uint32(np.uint32(1812433253) * (mt[i - 1] ^ (mt[i - 1] >> np.uint32(30))) + np.uint32(i))
    idx[0] = 624
    gauss[0] = 0.0
    gauss[1] = 0.0


@njit(cache=True)
def rk_double(mt, idx):
    a = np.float64(rk_next(mt, idx) >> np.uint32(5))
    b = np.float64(rk_next(mt, idx) >> np.uint32(6))
    return (a * 67108864.0 + b) / 9007199254740992.0


@njit(cache=True)
def rk_gauss(mt, idx, gauss):
    if gauss[0] != 0.0:
        t = gauss[1]
        gauss[0] = 0.0
        gauss[1] = 0.0
        return t
    while True:
        x1 = 2.0 * rk_double(mt, idx) - 1.0
        x2 = 2.0 * rk_double(mt, idx) - 1.0
        r2 = x1 * x1 + x2 * x2
        if r2 < 1.0 and r2 != 0.0:
            break
    f = np.sqrt(-2.0 * np.log(r2) / r2)
    gauss[0] = 1.0
    gauss[1] = f * x1
    return f * x2
