# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport log, sqrt
from libc.stdint cimport uint32_t


cdef class RNG:
    def __init__(self, unsigned int seed):
        cdef int i
        self.mt[0] = <uint32_t>seed
        for i in range(1, 624):
            self.mt[i] = <uint32_t>(<uint32_t>1812433253 *
                                    (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) +
                                    <uint32_t>i)
        self.idx = 624
        self.has_gauss = False
        self.gauss_val = 0.0

    cdef uint32_t next_u32(self):
        cdef uint32_t y, v
        cdef int i
        if self.idx >= 624:
            for i in range(624):
                y = ((self.mt[i] & <uint32_t>0x80000000) |
                     (self.mt[(i + 1) % 624] & <uint32_t>0x7FFFFFFF))
                v = self.mt[(i + 397) % 624] ^ (y >> 1)
                if y & <uint32_t>1:
                    v ^= <uint32_t>0x9908B0DF
                self.mt[i] = v
            self.idx = 0
        y = self.mt[self.idx]
        self.idx += 1
        y ^= y >> 11
        y ^= (y << 7) & <uint32_t>0x9D2C5680
        y ^= (y << 15) & <uint32_t>0xEFC60000
        y ^= y >> 18
        return y

    cdef double rk_double(self):
        cdef double a = <double>(self.next_u32() >> 5)
        cdef double b = <double>(self.next_u32() >> 6)
        return (a * 67108864.0 + b) / 9007199254740992.0

    cdef double rk_gauss(self):
        cdef double t, x1, x2, r2, f
        if self.has_gauss:
            t = self.gauss_val
            self.has_gauss = False
            self.gauss_val = 0.0
            return t
        while True:
            x1 = 2.0 * self.rk_double() - 1.0
            x2 = 2.0 * self.rk_double() - 1.0
            r2 = x1 * x1 + x2 * x2
            if r2 < 1.0 and r2 != 0.0:
                break
        f = sqrt(-2.0 * log(r2) / r2)
        self.has_gauss = True
        self.gauss_val = f * x1
        return f * x2
