from libc.stdint cimport uint32_t


cdef class RNG:
    cdef uint32_t mt[624]
    cdef int idx
    cdef bint has_gauss
    cdef double gauss_val

    cdef uint32_t next_u32(self)
    cdef double rk_double(self)
    cdef double rk_gauss(self)
