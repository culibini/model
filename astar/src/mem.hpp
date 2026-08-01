#pragma once
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>
#include <vector>
#ifdef __linux__
#include <sys/mman.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace astar {

template <typename T>
struct Buf {
    T* p = nullptr;
    size_t n = 0;
    size_t map_bytes = 0;

    static constexpr size_t BIG_THRESHOLD = 8u << 20;

    Buf() = default;
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    ~Buf() { release(); }

    void release() {
#ifdef __linux__
        if (map_bytes) {
            ::munmap((void*)p, map_bytes);
            p = nullptr; n = 0; map_bytes = 0;
            return;
        }
#endif
        std::free(p);
        p = nullptr; n = 0;
    }

    bool map_big(size_t bytes) {
#ifdef __linux__
        const size_t HP = (size_t)1 << 21;
        const size_t rounded = (bytes + HP - 1) / HP * HP;
        void* q = ::mmap(nullptr, rounded, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (q == MAP_FAILED) return false;
#ifdef MADV_HUGEPAGE
        ::madvise(q, rounded, MADV_HUGEPAGE);
#endif
        p = (T*)q;
        map_bytes = rounded;
        return true;
#else
        (void)bytes;
        return false;
#endif
    }

    void zeros(size_t count) {
        release();
        const size_t bytes = (count ? count : 1) * sizeof(T);
        if (bytes >= BIG_THRESHOLD && map_big(bytes)) { n = count; return; }
        p = (T*)std::calloc(count ? count : 1, sizeof(T));
        if (!p) throw std::bad_alloc();
        n = count;
    }
    void uninit(size_t count) {
        release();
        const size_t bytes = (count ? count : 1) * sizeof(T);
        if (bytes >= BIG_THRESHOLD && map_big(bytes)) { n = count; return; }
        p = (T*)std::malloc(bytes);
        if (!p) throw std::bad_alloc();
        n = count;
    }
    void filled(size_t count, T v) {
        uninit(count);
        std::fill(p, p + count, v);
    }
    void grow(size_t count) {
        T* old_p = p;
        size_t old_n = n;
        size_t old_map = map_bytes;
        p = nullptr; map_bytes = 0; n = 0;
        uninit(count);
        if (old_p) {
            std::memcpy(p, old_p, old_n * sizeof(T));
#ifdef __linux__
            if (old_map) { ::munmap((void*)old_p, old_map); old_p = nullptr; }
#endif
            std::free(old_p);
        }
    }

    inline T& operator[](size_t i) { return p[i]; }
    inline const T& operator[](size_t i) const { return p[i]; }
    inline T* data() { return p; }
    inline const T* data() const { return p; }
};

inline unsigned thread_count() {
#ifdef _OPENMP
    return (unsigned)omp_get_max_threads();
#else
    unsigned n = std::thread::hardware_concurrency();
    return n ? n : 1u;
#endif
}

template <typename F>
void parallel_for(size_t begin, size_t end, F&& body) {
    if (end <= begin) return;
    const size_t total = end - begin;
#ifdef _OPENMP
    if (total < 2) {
        for (size_t i = begin; i < end; ++i) body(i);
        return;
    }
    const long long n = (long long)total;
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) body(begin + (size_t)i);
#else
    unsigned nt = thread_count();
    if (nt <= 1 || total < 2) {
        for (size_t i = begin; i < end; ++i) body(i);
        return;
    }
    if ((size_t)nt > total) nt = (unsigned)total;
    const size_t chunk = (total + nt - 1) / nt;
    std::vector<std::thread> th;
    th.reserve(nt);
    for (unsigned t = 0; t < nt; ++t) {
        const size_t b = begin + (size_t)t * chunk;
        const size_t e = std::min(end, b + chunk);
        if (b >= e) break;
        th.emplace_back([&body, b, e] { for (size_t i = b; i < e; ++i) body(i); });
    }
    for (auto& x : th) x.join();
#endif
}
}
