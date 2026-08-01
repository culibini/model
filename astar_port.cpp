#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace cfg {

constexpr double LENGTH_WEIGHT         = 1.0;
constexpr double SAFETY_WEIGHT         = 5.0;
constexpr double SMOOTHNESS_WEIGHT     = 12.0;
constexpr double LEVEL_CHANGE_PENALTY  = 0.5;

constexpr double SAFETY_WEIGHT_COEFF   = 0.5;
constexpr double LENGTH_WEIGHT_COEFF   = 0.0;
constexpr double BASE_PATH_LENGTH      = 4000.0;

constexpr double OPTIMIZATION_RADIUS   = 400.0;
constexpr int    POINTS_TO_ADJUST      = 5;
constexpr int    NUM_CANDIDATES        = 60;
constexpr int    NUM_ITERATIONS        = 3;
constexpr int    LEVEL_OPTIMIZATION_RANGE = 3;

constexpr int    LOOKAHEAD_LEVELS      = 10;
constexpr double LEVEL_STAY_MULTIPLIER = 10.0;
constexpr int    NUM_LEVELS            = 32;

constexpr int    DIJKSTRA_STEP_SIZE    = 20;
constexpr double DIJKSTRA_LENGTH_PENALTY = 0.25;
constexpr double DIJKSTRA_DANGER_WEIGHT  = 2.0;
constexpr double DIJKSTRA_GOAL_TOLERANCE = 20.0;

constexpr double DANGER_SOFT_LIMIT       = 240.0;
constexpr double DANGER_HYPERBOLIC_RANGE = 60.0;
constexpr double DANGER_HYPERBOLIC_SCALE = 1000.0;

constexpr double FLIGHT_SPEED            = 1000.0;
constexpr double HOURS_PER_ARRAY_SLICE   = 1.0;

constexpr bool   HOUR_LOOKAHEAD_IS_NONE  = true;
constexpr int    HOUR_LOOKAHEAD_VALUE    = 0;
constexpr double HOUR_STAY_DISTANCE      = FLIGHT_SPEED * 0.25;
constexpr double HOUR_SWITCH_PENALTY     = 25.0;

constexpr long long MAX_NODES            = 35000000LL;

static const double LEVEL_PENALTIES[NUM_LEVELS] = {
    10, 10,  5,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  5,
    10, 10, 20, 20, 30, 30, 40, 40, 50, 50,
    60, 60
};

constexpr uint32_t RNG_SEED = 12345u;

}

static inline double py_round(double x) { return std::nearbyint(x); }
static inline long   py_round_i(double x) { return (long)std::nearbyint(x); }

static inline long   py_trunc_i(double x) { return (long)std::trunc(x); }

static inline double hypot2(double dx, double dy) {
    return std::sqrt(dx * dx + dy * dy);
}

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

    inline T& operator[](size_t i) { return p[i]; }
    inline const T& operator[](size_t i) const { return p[i]; }
    inline T* data() { return p; }
    inline const T* data() const { return p; }
};

static unsigned g_threads = 0;

static double g_wastar = 1.0;

static unsigned thread_count() {
    if (g_threads) return g_threads;
#ifdef _OPENMP
    return (unsigned)omp_get_max_threads();
#else
    unsigned n = std::thread::hardware_concurrency();
    return n ? n : 1u;
#endif
}

template <typename F>
static void parallel_for(size_t begin, size_t end, F&& body) {
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

struct FlatDangerMap {
    struct Slot {
        uint64_t key;
        double   danger;
        int32_t  hour;
    };
    static constexpr uint64_t EMPTY = ~0ull;
    static constexpr uint64_t TOMB  = ~0ull - 1;

    std::vector<Slot> slots;
    size_t mask = 0;
    size_t live = 0;
    size_t used = 0;

    FlatDangerMap() { rehash(1u << 20); }

    static inline uint64_t mix(uint64_t x) {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    }

    void rehash(size_t cap) {
        size_t c = 1;
        while (c < cap) c <<= 1;
        std::vector<Slot> old;
        old.swap(slots);
        slots.assign(c, Slot{EMPTY, 0.0, 0});
        mask = c - 1;
        live = 0;
        used = 0;
        for (const Slot& s : old)
            if (s.key != EMPTY && s.key != TOMB) put(s.key, s.danger, s.hour);
    }

    inline const Slot* get(uint64_t key) const {
        size_t i = mix(key) & mask;
        while (true) {
            const Slot& s = slots[i];
            if (s.key == key) return &s;
            if (s.key == EMPTY) return nullptr;
            i = (i + 1) & mask;
        }
    }

    void put(uint64_t key, double danger, int32_t hour) {
        if ((used + 1) * 10 >= slots.size() * 7) rehash(slots.size() * 2);
        size_t i = mix(key) & mask;
        size_t first_tomb = (size_t)-1;
        while (true) {
            Slot& s = slots[i];
            if (s.key == key) { s.danger = danger; s.hour = hour; return; }
            if (s.key == TOMB && first_tomb == (size_t)-1) first_tomb = i;
            if (s.key == EMPTY) {
                if (first_tomb != (size_t)-1) i = first_tomb;
                else used++;
                slots[i] = Slot{key, danger, hour};
                live++;
                return;
            }
            i = (i + 1) & mask;
        }
    }

    void erase(uint64_t key) {
        size_t i = mix(key) & mask;
        while (true) {
            Slot& s = slots[i];
            if (s.key == key) { s.key = TOMB; live--; return; }
            if (s.key == EMPTY) return;
            i = (i + 1) & mask;
        }
    }

    inline size_t size() const { return live; }
};

struct Timer {
    std::chrono::steady_clock::time_point t0;
    Timer() : t0(std::chrono::steady_clock::now()) {}
    double sec() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

static void log_info(const std::string& s)  { std::cout << "[INFO ] " << s << "\n"; }
static void log_warn(const std::string& s)  { std::cout << "[WARN ] " << s << "\n"; }
static void log_error(const std::string& s) { std::cout << "[ERROR] " << s << "\n"; }

struct NpyArray {
    std::vector<size_t> shape;
    std::vector<double> data;

    size_t size() const {
        size_t n = 1;
        for (size_t s : shape) n *= s;
        return n;
    }
};

static bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

static NpyArray load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("FileNotFoundError: " + path);

    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("ValueError: not a .npy file: " + path);

    uint8_t major = 0, minor = 0;
    f.read((char*)&major, 1);
    f.read((char*)&minor, 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t h16 = 0;
        f.read((char*)&h16, 2);
        header_len = h16;
    } else {
        f.read((char*)&header_len, 4);
    }

    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    auto grab_quoted = [&](const std::string& key) -> std::string {
        size_t k = header.find(key);
        if (k == std::string::npos) return "";
        size_t q1 = header.find_first_of("'\"", k + key.size());
        if (q1 == std::string::npos) return "";
        char qc = header[q1];
        size_t q2 = header.find(qc, q1 + 1);
        return header.substr(q1 + 1, q2 - q1 - 1);
    };

    std::string descr = grab_quoted("'descr'");
    if (descr.empty()) descr = grab_quoted("\"descr\"");

    bool fortran = header.find("'fortran_order': True") != std::string::npos ||
                   header.find("\"fortran_order\": True") != std::string::npos;
    if (fortran)
        throw std::runtime_error("ValueError: fortran_order .npy is not supported: " + path);

    NpyArray out;
    {
        size_t k = header.find("'shape'");
        if (k == std::string::npos) k = header.find("\"shape\"");
        size_t p1 = header.find('(', k);
        size_t p2 = header.find(')', p1);
        std::string s = header.substr(p1 + 1, p2 - p1 - 1);
        std::string num;
        for (char c : s) {
            if (std::isdigit((unsigned char)c)) num.push_back(c);
            else if (!num.empty()) { out.shape.push_back(std::stoul(num)); num.clear(); }
        }
        if (!num.empty()) out.shape.push_back(std::stoul(num));
    }

    size_t n = out.size();
    out.data.resize(n);

    std::string d = descr;
    if (!d.empty() && (d[0] == '<' || d[0] == '=' || d[0] == '|' || d[0] == '>')) {
        if (d[0] == '>' && d.size() > 2 && d[2] != '1')
            throw std::runtime_error("ValueError: big-endian .npy is not supported: " + path);
        d = d.substr(1);
    }

    auto read_all = [&](size_t item) {
        std::vector<char> raw(n * item);
        f.read(raw.data(), (std::streamsize)(n * item));
        if ((size_t)f.gcount() != n * item)
            throw std::runtime_error("ValueError: truncated .npy: " + path);
        return raw;
    };

    if (d == "f8") {
        auto raw = read_all(8);
        std::memcpy(out.data.data(), raw.data(), n * 8);
    } else if (d == "f4") {
        auto raw = read_all(4);
        const float* p = (const float*)raw.data();
        for (size_t i = 0; i < n; ++i) out.data[i] = (double)p[i];
    } else if (d == "u1") {
        auto raw = read_all(1);
        const uint8_t* p = (const uint8_t*)raw.data();
        for (size_t i = 0; i < n; ++i) out.data[i] = (double)p[i];
    } else if (d == "i1") {
        auto raw = read_all(1);
        const int8_t* p = (const int8_t*)raw.data();
        for (size_t i = 0; i < n; ++i) out.data[i] = (double)p[i];
    } else if (d == "u2") {
        auto raw = read_all(2);
        const uint16_t* p = (const uint16_t*)raw.data();
        for (size_t i = 0; i < n; ++i) out.data[i] = (double)p[i];
    } else if (d == "i2") {
        auto raw = read_all(2);
        const int16_t* p = (const int16_t*)raw.data();
        for (size_t i = 0; i < n; ++i) out.data[i] = (double)p[i];
    } else if (d == "u4") {
        auto raw = read_all(4);
        const uint32_t* p = (const uint32_t*)raw.data();
        for (size_t i = 0; i < n; ++i) out.data[i] = (double)p[i];
    } else if (d == "i4") {
        auto raw = read_all(4);
        const int32_t* p = (const int32_t*)raw.data();
        for (size_t i = 0; i < n; ++i) out.data[i] = (double)p[i];
    } else if (d == "u8") {
        auto raw = read_all(8);
        const uint64_t* p = (const uint64_t*)raw.data();
        for (size_t i = 0; i < n; ++i) out.data[i] = (double)p[i];
    } else if (d == "i8") {
        auto raw = read_all(8);
        const int64_t* p = (const int64_t*)raw.data();
        for (size_t i = 0; i < n; ++i) out.data[i] = (double)p[i];
    } else if (d == "b1") {
        auto raw = read_all(1);
        const uint8_t* p = (const uint8_t*)raw.data();
        for (size_t i = 0; i < n; ++i) out.data[i] = p[i] ? 1.0 : 0.0;
    } else {
        throw std::runtime_error("ValueError: unsupported dtype '" + descr + "' in " + path);
    }

    return out;
}

struct NumpyRandom {
    std::mt19937 mt;
    bool   has_gauss = false;
    double cached_gauss = 0.0;

    void seed(uint32_t s) { mt.seed(s); has_gauss = false; cached_gauss = 0.0; }

    double random() {
        uint32_t a = mt() >> 5;
        uint32_t b = mt() >> 6;
        return (a * 67108864.0 + b) / 9007199254740992.0;
    }

    double gauss() {
        if (has_gauss) {
            double t = cached_gauss;
            cached_gauss = 0.0;
            has_gauss = false;
            return t;
        }
        double x1, x2, r2;
        do {
            x1 = 2.0 * random() - 1.0;
            x2 = 2.0 * random() - 1.0;
            r2 = x1 * x1 + x2 * x2;
        } while (r2 >= 1.0 || r2 == 0.0);
        double f = std::sqrt(-2.0 * std::log(r2) / r2);
        cached_gauss = f * x1;
        has_gauss = true;
        return f * x2;
    }

    double normal(double loc, double scale) { return loc + scale * gauss(); }
};

static NumpyRandom g_rng;

static inline void map_to_3d_coords(double x, double y, double scale_x, double scale_y,
                                    int max_x3d, int max_y3d, int& x3d, int& y3d) {
    long xi = py_trunc_i(x / scale_x);
    long yi = py_trunc_i(y / scale_y);
    if (xi < 0) xi = 0;
    if (yi < 0) yi = 0;
    if (xi > max_x3d) xi = max_x3d;
    if (yi > max_y3d) yi = max_y3d;
    x3d = (int)xi;
    y3d = (int)yi;
}

struct Grid2D {
    int width = 0, height = 0;
    std::vector<double> v;
    inline double at(int y, int x) const { return v[(size_t)y * width + x]; }
    inline double& at(int y, int x) { return v[(size_t)y * width + x]; }
};

struct GridBool {
    int width = 0, height = 0;
    std::vector<uint8_t> v;
    inline uint8_t at(int y, int x) const { return v[(size_t)y * width + x]; }
};

static bool line_clear(double x1, double y1, double x2, double y2, const GridBool& valid) {
    long x1i = py_round_i(x1), y1i = py_round_i(y1);
    long x2i = py_round_i(x2), y2i = py_round_i(y2);
    long dx = std::labs(x2i - x1i);
    long dy = std::labs(y2i - y1i);
    long x = x1i, y = y1i;
    long sx = (x2i > x1i) ? 1 : -1;
    long sy = (y2i > y1i) ? 1 : -1;
    const int h = valid.height, w = valid.width;

    if (x < 0 || x >= w || y < 0 || y >= h) return false;

    if (dx > dy) {
        double err = dx / 2.0;
        while (x != x2i) {
            if (!valid.at((int)y, (int)x)) return false;
            err -= (double)dy;
            if (err < 0) { y += sy; err += (double)dx; }
            x += sx;
        }
    } else {
        double err = dy / 2.0;
        while (y != y2i) {
            if (!valid.at((int)y, (int)x)) return false;
            err -= (double)dx;
            if (err < 0) { x += sx; err += (double)dy; }
            y += sy;
        }
    }
    if (x2i < 0 || x2i >= w || y2i < 0 || y2i >= h) return false;
    return valid.at((int)y2i, (int)x2i) != 0;
}

static void generate_candidates(double center_x, double center_y, int num_candidates,
                                int max_attempts, double radius, double width, double height,
                                bool use_direction, double dir_x, double dir_y,
                                std::vector<double>& cx_out, std::vector<double>& cy_out) {
    cx_out.clear();
    cy_out.clear();
    int attempts = 0;
    while ((int)cx_out.size() < num_candidates && attempts < max_attempts) {
        attempts++;
        double angle, r;

        if (use_direction && g_rng.random() < 0.7) {
            double angle_variation = g_rng.normal(0.0, 0.5);
            double base_angle = std::atan2(dir_y, dir_x);
            angle = base_angle + angle_variation;
            r = g_rng.random() * radius;
        } else {
            angle = g_rng.random() * 2.0 * M_PI;
            r = g_rng.random() * radius;
        }
        double dx = r * std::cos(angle);
        double dy = r * std::sin(angle);
        double cx = center_x + dx;
        double cy = center_y + dy;
        cx = std::max(0.0, std::min(width - 1.0, cx));
        cy = std::max(0.0, std::min(height - 1.0, cy));
        cx_out.push_back(cx);
        cy_out.push_back(cy);
    }
}

static void smooth_adjust_neighbors(const std::vector<double>& path_xs,
                                    const std::vector<double>& path_ys,
                                    int center_index, int points_to_adjust,
                                    const GridBool& valid,
                                    const uint8_t* frozen,
                                    std::vector<double>& adj_xs,
                                    std::vector<double>& adj_ys) {
    const int n_points = (int)path_xs.size();
    adj_xs = path_xs;
    adj_ys = path_ys;
    const int h = valid.height, w = valid.width;

    int start_idx = std::max(1, center_index - points_to_adjust);
    int end_idx   = std::min(n_points - 2, center_index + points_to_adjust);

    for (int i = start_idx; i <= end_idx; ++i) {
        if (i == center_index) continue;
        if (frozen && frozen[i]) continue;

        int window_start = std::max(1, i - 1);
        int window_end   = std::min(n_points - 2, i + 1);
        int count = 0;
        double sum_x = 0.0, sum_y = 0.0;

        for (int j = window_start; j <= window_end; ++j) {
            sum_x += path_xs[j];
            sum_y += path_ys[j];
            count++;
        }

        if (count > 0) {
            double smoothed_x = sum_x / count;
            double smoothed_y = sum_y / count;
            long sx = py_round_i(smoothed_x);
            long sy = py_round_i(smoothed_y);

            if (sx >= 0 && sx < w && sy >= 0 && sy < h && valid.at((int)sy, (int)sx)) {
                if (line_clear(adj_xs[i - 1], adj_ys[i - 1], smoothed_x, smoothed_y, valid)) {
                    if (line_clear(smoothed_x, smoothed_y, adj_xs[i + 1], adj_ys[i + 1], valid)) {
                        adj_xs[i] = smoothed_x;
                        adj_ys[i] = smoothed_y;
                    }
                }
            }
        }
    }
}

static bool is_candidate_safe(const std::vector<double>& path_xs,
                              const std::vector<double>& path_ys,
                              int center_index, double candidate_x, double candidate_y,
                              int points_to_adjust, const GridBool& valid,
                              const uint8_t* frozen) {
    const int n_points = (int)path_xs.size();
    const int h = valid.height, w = valid.width;

    std::vector<double> temp_xs = path_xs;
    std::vector<double> temp_ys = path_ys;
    temp_xs[center_index] = candidate_x;
    temp_ys[center_index] = candidate_y;

    std::vector<double> sx, sy;
    smooth_adjust_neighbors(temp_xs, temp_ys, center_index, points_to_adjust, valid, frozen, sx, sy);
    temp_xs.swap(sx);
    temp_ys.swap(sy);

    int start_idx = std::max(0, center_index - points_to_adjust);
    int end_idx   = std::min(n_points - 1, center_index + points_to_adjust);

    for (int i = start_idx; i <= end_idx; ++i) {
        long xi = py_round_i(temp_xs[i]);
        long yi = py_round_i(temp_ys[i]);
        if (xi < 0 || xi >= w || yi < 0 || yi >= h) return false;
        if (!valid.at((int)yi, (int)xi)) return false;
    }
    for (int i = start_idx; i < end_idx; ++i) {
        if (!line_clear(temp_xs[i], temp_ys[i], temp_xs[i + 1], temp_ys[i + 1], valid))
            return false;
    }
    return true;
}

static bool is_point_and_neighbors_safe(const std::vector<double>& path_xs,
                                        const std::vector<double>& path_ys,
                                        int center_index, double point_x, double point_y,
                                        const GridBool& valid) {
    const int h = valid.height, w = valid.width;
    long px = py_round_i(point_x);
    long py = py_round_i(point_y);

    if (px < 0 || px >= w || py < 0 || py >= h) return false;
    if (!valid.at((int)py, (int)px)) return false;

    if (center_index > 0) {
        if (!line_clear(path_xs[center_index - 1], path_ys[center_index - 1],
                        point_x, point_y, valid))
            return false;
    }
    if (center_index < (int)path_xs.size() - 1) {
        if (!line_clear(point_x, point_y,
                        path_xs[center_index + 1], path_ys[center_index + 1], valid))
            return false;
    }
    return true;
}

static std::vector<double> recalculate_distances(const std::vector<double>& xs,
                                                 const std::vector<double>& ys) {
    const size_t n = xs.size();
    std::vector<double> out(n, 0.0);
    for (size_t i = 1; i < n; ++i)
        out[i] = out[i - 1] + hypot2(xs[i] - xs[i - 1], ys[i] - ys[i - 1]);
    return out;
}

static std::vector<double> level_distances(const std::vector<double>& xs,
                                           const std::vector<double>& ys,
                                           const std::vector<int>& levels) {
    const size_t n = xs.size();
    std::vector<double> out(n, 0.0);
    for (size_t i = 1; i < n; ++i) {
        if (levels[i] == levels[i - 1])
            out[i] = out[i - 1] + hypot2(xs[i] - xs[i - 1], ys[i] - ys[i - 1]);
        else
            out[i] = 0.0;
    }
    return out;
}

static std::vector<double> recompute_stay_requirements(const std::vector<int>& levels) {
    const size_t n = levels.size();
    std::vector<double> out(n, 0.0);
    for (size_t i = 1; i < n; ++i) {
        if (levels[i] > levels[i - 1])
            out[i] = cfg::LEVEL_STAY_MULTIPLIER * (levels[i] - levels[i - 1]);
        else
            out[i] = 0.0;
    }
    return out;
}

struct DangerMapCache {
    int width = 0, height = 0;
    Grid2D  danger_cache;
    GridBool validity_cache;

    explicit DangerMapCache(const Grid2D& danger_map_2d) {
        height = danger_map_2d.height;
        width  = danger_map_2d.width;
        danger_cache = apply_hyperbolic_penalty(danger_map_2d);
        validity_cache.width = width;
        validity_cache.height = height;
        validity_cache.v.assign((size_t)width * height, 1);

        double mb = (double)(validity_cache.v.size() * sizeof(uint8_t) +
                             danger_cache.v.size() * sizeof(double)) / 1024.0 / 1024.0;
        char buf[256];
        std::snprintf(buf, sizeof(buf), "Кеш создан: %dx%d, память: %.1f MB", width, height, mb);
        std::cout << buf << "\n";
    }

    static Grid2D apply_hyperbolic_penalty(const Grid2D& base_in) {
        Grid2D base = base_in;
        if (cfg::DANGER_HYPERBOLIC_RANGE <= 0.0) return base;

        bool any = false;
        for (double v : base.v) {
            if (v - cfg::DANGER_SOFT_LIMIT > 0.0) { any = true; break; }
        }
        if (!any) return base;

        const double cap = cfg::DANGER_HYPERBOLIC_RANGE * 0.999999;
        double* __restrict p = base.v.data();
        const int W = base.width;
        parallel_for(0, (size_t)base.height, [&](size_t row) {
            double* __restrict r = p + row * W;
            for (int i = 0; i < W; ++i) {
                double delta = r[i] - cfg::DANGER_SOFT_LIMIT;
                if (delta > 0.0) {
                    double capped = std::min(delta, cap);
                    double denom  = std::max(1e-6, cfg::DANGER_HYPERBOLIC_RANGE - capped);
                    r[i] += cfg::DANGER_HYPERBOLIC_SCALE * (capped / denom);
                }
            }
        });
        return base;
    }

    inline bool is_valid_fast(long x, long y) const {
        return !(x < 0 || x >= width || y < 0 || y >= height);
    }

    inline double get_danger_fast(long x, long y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) return 1e9;
        return danger_cache.at((int)y, (int)x);
    }
    inline bool get_cached_validity(double px, double py) const {
        return is_valid_fast(py_round_i(px), py_round_i(py));
    }
    inline double get_cached_danger(double px, double py) const {
        return get_danger_fast(py_round_i(px), py_round_i(py));
    }
};

static inline double distance_to_hour_index(double distance_traveled, double arrays_count,
                                            double flight_speed = cfg::FLIGHT_SPEED,
                                            double hours_per_array_slice = cfg::HOURS_PER_ARRAY_SLICE) {
    double time_hours = distance_traveled / flight_speed;
    long hour_idx = py_trunc_i(time_hours / hours_per_array_slice);
    if (hour_idx < 0) return 0.0;
    double max_idx = arrays_count - 1.0;
    return ((double)hour_idx <= max_idx) ? (double)hour_idx : max_idx;
}

static inline int get_hour_lookahead_value(int max_hours) {
    if (cfg::HOUR_LOOKAHEAD_IS_NONE) return max_hours > 0 ? max_hours - 1 : 0;
    return std::max(0, std::min(cfg::HOUR_LOOKAHEAD_VALUE, max_hours - 1));
}

struct Array3D {
    int levels = 0, height = 0, width = 0;
    std::vector<double> v;
    inline double at(int l, int y, int x) const {
        return v[((size_t)l * height + y) * width + x];
    }
};

struct DangerValueCache {
    const DangerMapCache* map_cache = nullptr;
    const std::vector<Array3D>* arrays_3d = nullptr;
    double scale_x = 1.0, scale_y = 1.0, flight_speed = cfg::FLIGHT_SPEED;
    int max_x3d = 0, max_y3d = 0;

    FlatDangerMap cache;
    std::vector<uint64_t> insertion_order;
    size_t order_head = 0;

    DangerValueCache(const DangerMapCache* mc, const std::vector<Array3D>* a3d,
                     double sx, double sy, double fs)
        : map_cache(mc), arrays_3d(a3d), scale_x(sx), scale_y(sy), flight_speed(fs) {
        if (!a3d->empty()) {
            max_y3d = (*a3d)[0].height - 1;
            max_x3d = (*a3d)[0].width - 1;
        } else {
            max_y3d = 0;
            max_x3d = 0;
        }
    }

    static inline uint64_t make_key(long x, long y, int level, int hour) {

        return (((uint64_t)(uint32_t)x << 40) | ((uint64_t)(uint32_t)y << 20) |
                ((uint64_t)(uint32_t)level << 8) | (uint64_t)(uint32_t)hour);
    }

    inline double get_level_penalty_cached(int level) const {
        if (level >= 0 && level < cfg::NUM_LEVELS) return cfg::LEVEL_PENALTIES[level];
        return 0.0;
    }

    std::pair<double, int> get_total_danger_cached(double px, double py, int level,
                                                   double distance_traveled) {

        long x = py_trunc_i(px);
        long y = py_trunc_i(py);
        int hour_idx = (int)distance_to_hour_index(distance_traveled,
                                                   (double)arrays_3d->size(), flight_speed);
        uint64_t key = make_key(x, y, level, hour_idx);

        if (const FlatDangerMap::Slot* s = cache.get(key))
            return std::pair<double, int>(s->danger, s->hour);

        double base_danger = map_cache->get_cached_danger(px, py);
        double level_penalty = get_level_penalty_cached(level);

        int x3d, y3d;
        map_to_3d_coords(px, py, scale_x, scale_y, max_x3d, max_y3d, x3d, y3d);

        double level_danger = 0.0;
        if (hour_idx >= 0 && hour_idx < (int)arrays_3d->size()) {
            const Array3D& arr = (*arrays_3d)[hour_idx];
            if (level >= 0 && level < arr.levels) level_danger = arr.at(level, y3d, x3d);
        }

        std::pair<double, int> result(base_danger + level_danger + level_penalty, hour_idx);
        cache.put(key, result.first, (int32_t)result.second);
        insertion_order.push_back(key);

        if (cache.size() > 500000) {
            size_t to_drop = std::min<size_t>(50000, insertion_order.size() - order_head);
            for (size_t i = 0; i < to_drop; ++i) cache.erase(insertion_order[order_head + i]);
            order_head += to_drop;
            if (order_head > insertion_order.size() / 2) {
                insertion_order.erase(insertion_order.begin(),
                                      insertion_order.begin() + order_head);
                order_head = 0;
            }
        }
        return result;
    }

    int find_best_level_for_point(double px, double py, double distance_traveled) {
        int best_level = 0;
        double best_danger = std::numeric_limits<double>::infinity();
        for (int level = 0; level < cfg::NUM_LEVELS; ++level) {
            double d = get_total_danger_cached(px, py, level, distance_traveled).first;
            if (d < best_danger) { best_danger = d; best_level = level; }
        }
        return best_level;
    }
};

struct Environment {
    DangerMapCache*   map2_cache = nullptr;
    DangerValueCache* danger_cache = nullptr;
    std::vector<Array3D> arrays_3d;
    Grid2D raw_map;
    int height = 0, width = 0;
    int array_levels = 0, array_height = 0, array_width = 0;
    int MAX_X_3D = 0, MAX_Y_3D = 0;
    double SCALE_X = 1.0, SCALE_Y = 1.0;

    Environment(const std::string& map_file, const std::vector<std::string>& array_files) {
        NpyArray m = load_npy(map_file);
        if (m.shape.size() != 2)
            throw std::runtime_error("ValueError: map must be 2D, got ndim=" +
                                     std::to_string(m.shape.size()));
        raw_map.height = (int)m.shape[0];
        raw_map.width  = (int)m.shape[1];
        raw_map.v      = m.data;

        height = raw_map.height;
        width  = raw_map.width;

        map2_cache = new DangerMapCache(raw_map);

        for (const auto& p : array_files) {
            if (!file_exists(p)) continue;
            NpyArray a = load_npy(p);
            if (a.shape.size() != 3)
                throw std::runtime_error("ValueError: 3D array expected: " + p);
            Array3D arr;
            arr.levels = (int)a.shape[0];
            arr.height = (int)a.shape[1];
            arr.width  = (int)a.shape[2];
            arr.v      = std::move(a.data);
            arrays_3d.push_back(std::move(arr));
        }
        if (arrays_3d.empty())
            throw std::runtime_error("FileNotFoundError: No .npy arrays found for 3D risk data");

        array_levels = arrays_3d[0].levels;
        array_height = arrays_3d[0].height;
        array_width  = arrays_3d[0].width;
        MAX_X_3D = array_width - 1;
        MAX_Y_3D = array_height - 1;
        SCALE_Y = (double)height / (double)array_height;
        SCALE_X = (double)width  / (double)array_width;

        danger_cache = new DangerValueCache(map2_cache, &arrays_3d, SCALE_X, SCALE_Y,
                                            cfg::FLIGHT_SPEED);
    }

    ~Environment() { delete danger_cache; delete map2_cache; }
};

struct HeapPayload {
    double   cost;
    uint32_t state;
};

struct HeapNode {
    double   priority;
    double   cost;
    uint32_t state;
};

struct Heap {
    Buf<double>      pri;
    Buf<HeapPayload> pay;
    long long size = 0, capacity = 0;

    void init(long long cap) {
        capacity = cap;
        pri.uninit((size_t)cap);
        pay.uninit((size_t)cap);
        size = 0;
    }

    void push(double priority, double cost, uint32_t state) {
        double*      __restrict p  = pri.data();
        HeapPayload* __restrict pl = pay.data();
        long long idx = size;
        p[idx]  = priority;
        pl[idx] = HeapPayload{cost, state};
        while (idx > 0) {
            const long long parent = (idx - 1) / 2;
            if (p[parent] <= p[idx]) break;
            std::swap(p[parent], p[idx]);
            std::swap(pl[parent], pl[idx]);
            idx = parent;
        }
        size += 1;
    }

    HeapNode pop() {
        double*      __restrict p  = pri.data();
        HeapPayload* __restrict pl = pay.data();
        if (size == 0) return HeapNode{0.0, 0.0, 0};
        const double      top_p  = p[0];
        const HeapPayload top_pl = pl[0];
        size -= 1;
        p[0]  = p[size];
        pl[0] = pl[size];
        long long idx = 0;
        while (true) {
            const long long left = 2 * idx + 1, right = 2 * idx + 2;
            long long smallest = idx;
            if (left  < size && p[left]  < p[smallest]) smallest = left;
            if (right < size && p[right] < p[smallest]) smallest = right;
            if (smallest == idx) break;
            std::swap(p[idx], p[smallest]);
            std::swap(pl[idx], pl[smallest]);
            idx = smallest;
        }
        return HeapNode{top_p, top_pl.cost, top_pl.state};
    }
};

struct StateAux { double stay, hour_stay, total_dist; };
constexpr uint32_t NO_PREV = 0xFFFFFFFFu;

struct DijkstraArena {
    bool   ready = false;
    size_t n4 = 0;
    Buf<double>   danger_grid, min_nb_danger;
    Buf<uint8_t>  has_nb;
    Buf<double>   dist_grid;
    Buf<StateAux> aux;
    Buf<uint32_t> prev_state;
    Buf<uint8_t>  visited;
    Heap heap;

    Buf<double> min_dh, hgrid;
};

struct DijkstraRaw {
    std::vector<double> px, py;
    std::vector<int>    levels;
    std::vector<double> stay;
    std::vector<double> total_dist;
    std::vector<int>    arrays;
    std::vector<uint8_t> frozen;
    bool ok = false;
};

static void materialize_loiter_loops(DijkstraRaw& r,
                                     const double* danger_grid,
                                     const GridBool& validity,
                                     int grid_w, int grid_h,
                                     int width, int height,
                                     int step_size, int num_levels, int max_hours,
                                     double flight_speed) {
    const size_t n = r.px.size();
    r.frozen.assign(n, 0);
    if (n < 2) return;

    const size_t BLOCK = (size_t)num_levels * max_hours;
    bool any = false;
    for (size_t i = 0; i + 1 < n && !any; ++i) {
        const double geo = hypot2(r.px[i + 1] - r.px[i], r.py[i + 1] - r.py[i]);
        if (r.total_dist[i + 1] - r.total_dist[i] - geo > 1.0) any = true;
    }
    if (!any) return;

    std::vector<double> px, py;
    std::vector<int> lvl;
    std::vector<uint8_t> fz;
    px.push_back(r.px[0]); py.push_back(r.py[0]);
    lvl.push_back(r.levels[0]); fz.push_back(0);

    const int dx8[8] = {0, 1, 0, -1, 1, 1, -1, -1};
    const int dy8[8] = {1, 0, -1, 0, 1, -1, 1, -1};
    long total_loop_points = 0;

    for (size_t i = 0; i + 1 < n; ++i) {
        const double ax = r.px[i], ay = r.py[i];
        const double bx = r.px[i + 1], by = r.py[i + 1];
        const double geo = hypot2(bx - ax, by - ay);
        const double added = (r.total_dist[i + 1] - r.total_dist[i]) - geo;
        const int L = r.levels[i + 1];

        if (added <= 1.0) {
            px.push_back(bx); py.push_back(by); lvl.push_back(L); fz.push_back(0);
            continue;
        }

        px.push_back(bx); py.push_back(by); lvl.push_back(L); fz.push_back(1);

        const int bgx = (int)py_round_i(bx / (double)step_size);
        const int bgy = (int)py_round_i(by / (double)step_size);

        double dpos = r.total_dist[i] + geo;
        double rem  = added;
        while (rem > 1.0) {
            int h = (int)(dpos / flight_speed);
            if (h > max_hours - 1) h = max_hours - 1;
            double chunk = (h >= max_hours - 1)
                ? rem
                : std::min(rem, (double)(h + 1) * flight_speed - dpos);
            if (chunk <= 0.0) chunk = rem;

            double best = 1e18, nxr = ax, nyr = ay;
            for (int d = 0; d < 8; ++d) {
                const int gx2 = bgx + dx8[d], gy2 = bgy + dy8[d];
                if (gx2 < 0 || gy2 < 0 || gx2 >= grid_w || gy2 >= grid_h) continue;
                const int rx = gx2 * step_size, ry = gy2 * step_size;
                if (rx >= width || ry >= height) continue;
                if (!validity.at(ry, rx)) continue;
                const double dv = danger_grid[((size_t)gy2 * grid_w + gx2) * BLOCK +
                                              (size_t)L * max_hours + h];
                if (dv < best) { best = dv; nxr = rx; nyr = ry; }
            }

            const double loop_len = 2.0 * hypot2(nxr - bx, nyr - by);
            long loops = (long)std::ceil(chunk / loop_len);
            if (loops < 1) loops = 1;
            for (long k = 0; k < loops; ++k) {
                px.push_back(nxr); py.push_back(nyr); lvl.push_back(L); fz.push_back(1);
                px.push_back(bx);  py.push_back(by);  lvl.push_back(L); fz.push_back(1);
                total_loop_points += 2;
            }
            const double flown = (double)loops * loop_len;
            dpos += flown;
            rem  -= flown;
        }
    }

    const size_t m = px.size();
    std::vector<double> td(m, 0.0), st(m, 0.0);
    std::vector<int> ar(m, 0);
    for (size_t i = 1; i < m; ++i) {
        td[i] = td[i - 1] + hypot2(px[i] - px[i - 1], py[i] - py[i - 1]);
        int h = (int)(td[i] / flight_speed);
        ar[i] = h > max_hours - 1 ? max_hours - 1 : h;
        st[i] = (lvl[i] > lvl[i - 1])
            ? cfg::LEVEL_STAY_MULTIPLIER * (lvl[i] - lvl[i - 1]) : 0.0;
    }

    {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
                      "Ожидания материализованы: %zu -> %zu точек (+%ld точек петель)",
                      n, m, total_loop_points);
        log_info(buf);
    }

    r.px = std::move(px);
    r.py = std::move(py);
    r.levels = std::move(lvl);
    r.stay = std::move(st);
    r.total_dist = std::move(td);
    r.arrays = std::move(ar);
    r.frozen = std::move(fz);
}

static DijkstraRaw dijkstra_numba_grid(double start_x, double start_y,
                                       double goal_x, double goal_y,
                                       int start_level, int goal_level,
                                       const GridBool& validity_cache,
                                       const Grid2D& danger_cache_2d,
                                       const std::vector<Array3D>& arrays_3d_list,
                                       const double* level_penalties,
                                       int step_size, int lookahead_levels,
                                       double level_stay_multiplier, int num_levels,
                                       double flight_speed, int max_hours,
                                       double danger_weight, double length_penalty,
                                       double goal_tolerance, long long max_nodes,
                                       double scale_x, double scale_y,
                                       int max_x3d, int max_y3d,
                                       int hour_lookahead, double hour_stay_distance,
                                       double hour_switch_penalty,
                                       DijkstraArena& arena) {
    DijkstraRaw out;

    const int height = validity_cache.height;
    const int width  = validity_cache.width;
    const int grid_w = (int)std::ceil((double)width  / step_size) + 1;
    const int grid_h = (int)std::ceil((double)height / step_size) + 1;

    const int start_gx = (int)py_round_i(start_x / (double)step_size);
    const int start_gy = (int)py_round_i(start_y / (double)step_size);

    const size_t N4 = (size_t)grid_h * grid_w * num_levels * max_hours;
    auto IDX = [&](int gy, int gx, int l, int h) -> size_t {
        return (((size_t)gy * grid_w + gx) * num_levels + l) * max_hours + h;
    };

    {
        double gb = (double)N4 * (8.0 * 5 + 4.0 * 4 + 1.0) / 1024.0 / 1024.0 / 1024.0;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "Сетка поиска: %dx%d x %d уровней x %d часов = %.2fM состояний, ~%.2f GB",
                      grid_w, grid_h, num_levels, max_hours, (double)N4 / 1e6, gb);
        log_info(buf);
    }

    const size_t BLOCK = (size_t)num_levels * max_hours;

    if (N4 > 0xFFFFFFFFull) {

        log_error("Search space exceeds 2^32 states");
        out.ok = false;
        return out;
    }

    const bool arena_fresh = !arena.ready || arena.n4 != N4;

    Buf<double>&   danger_grid   = arena.danger_grid;
    Buf<double>&   min_nb_danger = arena.min_nb_danger;
    Buf<uint8_t>&  has_nb        = arena.has_nb;

    if (arena_fresh) {

    danger_grid.zeros(N4);

    parallel_for(0, (size_t)grid_h, [&](size_t gy_) {
        const int gy = (int)gy_;
        for (int gx = 0; gx < grid_w; ++gx) {
            const int nx_real = gx * step_size;
            const int ny_real = gy * step_size;
            if (nx_real >= width || ny_real >= height) continue;

            int x3d, y3d;
            map_to_3d_coords((double)nx_real, (double)ny_real, scale_x, scale_y,
                             max_x3d, max_y3d, x3d, y3d);
            const double base_danger = danger_cache_2d.at(ny_real, nx_real);
            double* __restrict blk = danger_grid.data() + IDX(gy, gx, 0, 0);

            for (int level = 0; level < num_levels; ++level) {
                const double lp = level_penalties[level];
                double* __restrict row = blk + (size_t)level * max_hours;
                for (int hour = 0; hour < max_hours; ++hour) {
                    const Array3D& arr = arrays_3d_list[hour];
                    double level_danger = (level < arr.levels) ? arr.at(level, y3d, x3d) : 0.0;
                    row[hour] = base_danger + level_danger + lp;
                }
            }
        }
    });

    min_nb_danger.zeros(N4);
    has_nb.zeros((size_t)grid_h * grid_w);

    {
        struct D8 { int dx, dy; };
        const D8 d8[8] = {{0,1},{1,0},{0,-1},{-1,0},{1,1},{1,-1},{-1,1},{-1,-1}};

        parallel_for(0, (size_t)grid_h, [&](size_t gy_) {
            const int gy = (int)gy_;
            const double* nb_base[8];
            for (int gx = 0; gx < grid_w; ++gx) {
                int nnb = 0;
                for (int b = 0; b < 8; ++b) {
                    const int bx = gx + d8[b].dx, by = gy + d8[b].dy;
                    if (bx < 0 || by < 0 || bx >= grid_w || by >= grid_h) continue;
                    const int bx_real = bx * step_size, by_real = by * step_size;
                    if (bx_real < 0 || by_real < 0 || bx_real >= width || by_real >= height) continue;
                    if (!validity_cache.at(by_real, bx_real)) continue;
                    nb_base[nnb++] = danger_grid.data() + IDX(by, bx, 0, 0);
                }
                if (nnb == 0) continue;
                has_nb[(size_t)gy * grid_w + gx] = 1;
                double* __restrict out = min_nb_danger.data() + IDX(gy, gx, 0, 0);
                const double* __restrict n0 = nb_base[0];
                for (size_t k = 0; k < BLOCK; ++k) out[k] = n0[k];
                for (int j = 1; j < nnb; ++j) {
                    const double* __restrict nj = nb_base[j];
                    for (size_t k = 0; k < BLOCK; ++k)
                        out[k] = out[k] < nj[k] ? out[k] : nj[k];
                }
            }
        });
    }
    }

    const double inf = 1e18;

    Buf<double>&   dist_grid  = arena.dist_grid;
    Buf<StateAux>& aux        = arena.aux;
    Buf<uint32_t>& prev_state = arena.prev_state;
    Buf<uint8_t>&  visited    = arena.visited;
    Heap&          heap       = arena.heap;

    const long long max_heap_nodes = (long long)N4;
    const long long heap_capacity = std::min(max_nodes, max_heap_nodes) + 10;

    if (arena_fresh) {
        dist_grid.filled(N4, inf);
        aux.zeros(N4);
        prev_state.filled(N4, NO_PREV);
        visited.zeros(N4);
        heap.init(heap_capacity);
        arena.n4 = N4;
        arena.ready = true;
    } else {

        std::fill(dist_grid.data(), dist_grid.data() + N4, inf);
        std::memset(aux.data(), 0, N4 * sizeof(StateAux));
        std::memset(prev_state.data(), 0xFF, N4 * sizeof(uint32_t));
        std::memset(visited.data(), 0, N4);
        heap.size = 0;
    }

    const size_t N2 = (size_t)grid_h * grid_w;
    {
        if (arena_fresh || arena.min_dh.n != N2) {

            arena.min_dh.uninit(N2);
            double* __restrict md = arena.min_dh.data();
            parallel_for(0, (size_t)grid_h, [&](size_t gy_) {
                for (int gx = 0; gx < grid_w; ++gx) {
                    const double* __restrict blk =
                        danger_grid.data() + ((size_t)gy_ * grid_w + gx) * BLOCK;
                    double m = blk[0];
                    for (size_t k = 1; k < BLOCK; ++k)
                        if (blk[k] < m) m = blk[k];
                    md[gy_ * grid_w + gx] = m;
                }
            });
        }

        arena.hgrid.filled(N2, 1e18);
        double* __restrict hg = arena.hgrid.data();
        const double* __restrict md = arena.min_dh.data();

        using QN = std::pair<double, int>;
        std::priority_queue<QN, std::vector<QN>, std::greater<QN>> pq;

        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                const double rx = (double)gx * step_size, ry = (double)gy * step_size;
                if (rx >= width || ry >= height) continue;
                if (std::fabs(rx - goal_x) <= goal_tolerance &&
                    std::fabs(ry - goal_y) <= goal_tolerance) {
                    hg[(size_t)gy * grid_w + gx] = 0.0;
                    pq.emplace(0.0, gy * grid_w + gx);
                }
            }
        }

        const double S2h = std::sqrt(2.0);
        const int hdx[8] = {0, 1, 0, -1, 1, 1, -1, -1};
        const int hdy[8] = {1, 0, -1, 0, 1, -1, 1, -1};
        while (!pq.empty()) {
            const QN top2 = pq.top();
            pq.pop();
            const int c = top2.second;
            if (top2.first > hg[c]) continue;
            const int cgx = c % grid_w, cgy = c / grid_w;

            const double enter_c = md[c] * danger_weight;
            for (int d = 0; d < 8; ++d) {
                const int agx = cgx + hdx[d], agy = cgy + hdy[d];
                if (agx < 0 || agy < 0 || agx >= grid_w || agy >= grid_h) continue;
                const int arx = agx * step_size, ary = agy * step_size;
                if (arx >= width || ary >= height) continue;
                if (!validity_cache.at(ary, arx)) continue;
                const double mul = (d < 4) ? 1.0 : S2h;
                const double cand =
                    top2.first + enter_c + mul * step_size * length_penalty;
                const size_t ai = (size_t)agy * grid_w + agx;
                if (cand < hg[ai]) {
                    hg[ai] = cand;
                    pq.emplace(cand, (int)ai);
                }
            }
        }
    }
    const double* __restrict hgrid_ptr = arena.hgrid.data();

    const int start_hour = 0;
    dist_grid[IDX(start_gy, start_gx, start_level, start_hour)] = 0.0;
    heap.push(0.0, 0.0, (uint32_t)IDX(start_gy, start_gx, start_level, start_hour));

    struct Dir { int dx, dy; double mul; };
    const double S2 = std::sqrt(2.0);
    const Dir directions[8] = {
        {0, 1, 1.0}, {1, 0, 1.0}, {0, -1, 1.0}, {-1, 0, 1.0},
        {1, 1, S2},  {1, -1, S2}, {-1, 1, S2},  {-1, -1, S2}
    };

    long long nodes_processed = 0;
    long long pushes_dropped = 0;
    bool goal_found = false;
    int gs_gx = start_gx, gs_gy = start_gy, gs_l = start_level, gs_h = start_hour;

    while (heap.size > 0 && nodes_processed < max_nodes) {
        const HeapNode top = heap.pop();
        const double cur_cost = top.cost;

        const size_t cur_ni = (size_t)top.state;
        const int hour_idx = (int)(cur_ni % (size_t)max_hours);
        size_t rest = cur_ni / (size_t)max_hours;
        const int level = (int)(rest % (size_t)num_levels);
        rest /= (size_t)num_levels;
        const int gx = (int)(rest % (size_t)grid_w);
        const int gy = (int)(rest / (size_t)grid_w);

        if (visited[cur_ni]) continue;
        visited[cur_ni] = 1;
        nodes_processed++;

        const double real_x = (double)gx * step_size;
        const double real_y = (double)gy * step_size;

        if (std::fabs(real_x - goal_x) <= goal_tolerance &&
            std::fabs(real_y - goal_y) <= goal_tolerance &&
            level == goal_level) {
            goal_found = true;
            gs_gx = gx; gs_gy = gy; gs_l = level; gs_h = hour_idx;
            break;
        }

        const StateAux cur_aux = aux[cur_ni];
        const double current_total_distance = cur_aux.total_dist;
        const double current_stay           = cur_aux.stay;
        const double current_hour_stay      = cur_aux.hour_stay;

        int nb_cnt = 0;
        int    nb_dir[8];
        size_t nb_block[8];
        for (int d = 0; d < 8; ++d) {
            const int ngx = gx + directions[d].dx;
            const int ngy = gy + directions[d].dy;
            if (ngx < 0 || ngy < 0 || ngx >= grid_w || ngy >= grid_h) continue;
            const int nx_real = ngx * step_size;
            const int ny_real = ngy * step_size;
            if (nx_real < 0 || ny_real < 0 || nx_real >= width || ny_real >= height) continue;
            if (!validity_cache.at(ny_real, nx_real)) continue;
            nb_dir[nb_cnt] = d;
            nb_block[nb_cnt] = ((size_t)ngy * grid_w + ngx) * BLOCK;
            nb_cnt++;
        }

        {
            const size_t centre = (size_t)level * max_hours;
            for (int k = 0; k < nb_cnt; ++k) {
                __builtin_prefetch(dist_grid.data()   + nb_block[k] + centre, 0, 1);
                __builtin_prefetch(danger_grid.data() + nb_block[k] + centre, 0, 1);
            }
        }

        for (int k = 0; k < nb_cnt; ++k) {
            const int d = nb_dir[k];
            const int ngx = gx + directions[d].dx;
            const int ngy = gy + directions[d].dy;

            const double step_distance = directions[d].mul * step_size;

            const double length_cost = step_distance * length_penalty;

            const double h_val = hgrid_ptr[(size_t)ngy * grid_w + ngx] * g_wastar;
            const bool   nb_ok = has_nb[(size_t)ngy * grid_w + ngx] != 0;

            const size_t nblock = nb_block[k];
            const double* __restrict dg_blk = danger_grid.data() + nblock;
            const double* __restrict mn_blk = min_nb_danger.data() + nblock;

            int lvl_lo = -lookahead_levels;
            if (level + lvl_lo < 0) lvl_lo = -level;
            int lvl_hi = (current_stay > 0.0) ? 0 : lookahead_levels;
            if (level + lvl_hi >= num_levels) lvl_hi = num_levels - 1 - level;

            for (int lvl_shift = lvl_lo; lvl_shift <= lvl_hi; ++lvl_shift) {
                const int next_level = level + lvl_shift;

                double new_stay;
                if (lvl_shift > 0)      new_stay = level_stay_multiplier * lvl_shift;
                else if (lvl_shift < 0) new_stay = 0.0;
                else {
                    new_stay = current_stay - step_distance;
                    if (new_stay < 0.0) new_stay = 0.0;
                }

                const double level_change_cost =
                    std::abs(lvl_shift) * cfg::LEVEL_CHANGE_PENALTY;
                const size_t lvl_off = (size_t)next_level * max_hours;

                double new_total_distance = current_total_distance + step_distance;

                int base_hour = (int)py_trunc_i(new_total_distance / flight_speed);
                if (base_hour < 0) base_hour = 0;
                if (base_hour >= max_hours) base_hour = max_hours - 1;

                const int hs_begin = (current_hour_stay > 0.0)
                    ? 0
                    : (hour_idx > base_hour ? hour_idx - base_hour : 0);
                const int hs_end = (current_hour_stay > 0.0) ? 0 : hour_lookahead;

                for (int hour_shift = hs_begin; hour_shift <= hs_end; ++hour_shift) {
                    int target_hour = base_hour + hour_shift;
                    if (target_hour >= max_hours) target_hour = max_hours - 1;

                    if (target_hour < hour_idx) continue;

                    double new_hour_stay;
                    if (target_hour > hour_idx) {
                        new_hour_stay = hour_stay_distance * (target_hour - hour_idx);
                    } else {
                        new_hour_stay = current_hour_stay - step_distance;
                        if (new_hour_stay < 0.0) new_hour_stay = 0.0;
                    }

                    const double total_danger = dg_blk[lvl_off + target_hour];

                    const double danger_cost = total_danger * danger_weight;
                    const double hour_change_cost =
                        hour_switch_penalty * (target_hour - hour_idx);

                    double added_cost = 0.0;
                    bool   loitered = false;

                    if (target_hour > base_hour) {
                        const double required_distance = (double)target_hour * flight_speed;
                        const double need_extra = required_distance - new_total_distance;
                        if (need_extra > 0.0 && nb_ok) {
                            loitered = true;
                            double add_c = 0.0;
                            double dpos  = new_total_distance;
                            double rem   = need_extra;
                            while (rem > 1e-9) {
                                int h = (int)(dpos / flight_speed);
                                if (h > max_hours - 1) h = max_hours - 1;
                                double chunk = (h >= max_hours - 1)
                                    ? rem
                                    : std::min(rem, (double)(h + 1) * flight_speed - dpos);
                                if (chunk <= 0.0) chunk = rem;
                                const double avg_h =
                                    0.5 * (dg_blk[lvl_off + h] + mn_blk[lvl_off + h]);
                                add_c += chunk * (avg_h * danger_weight / step_distance
                                                  + length_penalty);
                                dpos += chunk;
                                rem  -= chunk;
                            }
                            added_cost = add_c;
                            new_total_distance += need_extra;
                        }
                    }

                    const double step_cost = danger_cost + length_cost + level_change_cost +
                                             hour_change_cost + added_cost;
                    const double new_cost = cur_cost + step_cost;

                    const size_t ni = nblock + lvl_off + target_hour;
                    if (new_cost < dist_grid[ni]) {
                        dist_grid[ni] = new_cost;
                        aux[ni] = StateAux{new_stay, new_hour_stay, new_total_distance};
                        prev_state[ni] = (uint32_t)cur_ni;

                        if (heap.size < heap.capacity)
                            heap.push(new_cost + h_val, new_cost, (uint32_t)ni);
                        else
                            pushes_dropped++;
                    }

                    if (!loitered && target_hour == max_hours - 1 &&
                        base_hour + hour_shift >= max_hours - 1) break;
                }
            }
        }
    }

    if (!goal_found) {
        double best_cost = inf;
        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                const int lvl = goal_level;
                for (int hr = 0; hr < max_hours; ++hr) {
                    if (!visited[IDX(gy, gx, lvl, hr)]) continue;
                    const double real_x = (double)gx * step_size;
                    const double real_y = (double)gy * step_size;
                    if (std::fabs(real_x - goal_x) <= goal_tolerance * 3 &&
                        std::fabs(real_y - goal_y) <= goal_tolerance * 3) {
                        const double c = dist_grid[IDX(gy, gx, lvl, hr)];
                        if (c < best_cost) {
                            best_cost = c;
                            gs_gx = gx; gs_gy = gy; gs_l = lvl; gs_h = hr;
                        }
                    }
                }
            }
        }
        if (best_cost < inf) goal_found = true;
    }

    {
        char buf[240];
        std::snprintf(buf, sizeof(buf), "Обработано узлов: %lld", nodes_processed);
        log_info(buf);
        if (pushes_dropped > 0) {
            std::snprintf(buf, sizeof(buf),
                          "ВНИМАНИЕ: куча переполнялась, отброшено %lld пушей — "
                          "поиск мог потерять оптимальность (поведение унаследовано "
                          "от питона)", pushes_dropped);
            log_warn(buf);
        }
    }

    if (!goal_found) { out.ok = false; return out; }

    {

        char buf[160];
        std::snprintf(buf, sizeof(buf), "Стоимость цели: %.6f",
                      dist_grid[IDX(gs_gy, gs_gx, gs_l, gs_h)]);
        log_info(buf);
    }

    std::vector<double> ppx, ppy, pstay, pdist;
    std::vector<int> plvl, parr;

    size_t cur = IDX(gs_gy, gs_gx, gs_l, gs_h);
    while (true) {
        const int hr   = (int)(cur % (size_t)max_hours);
        const int lvl  = (int)((cur / max_hours) % (size_t)num_levels);
        const size_t c = cur / ((size_t)max_hours * num_levels);
        const int gx   = (int)(c % (size_t)grid_w);
        const int gy   = (int)(c / (size_t)grid_w);

        ppx.push_back((double)gx * step_size);
        ppy.push_back((double)gy * step_size);
        plvl.push_back(lvl);
        pstay.push_back(aux[cur].stay);
        pdist.push_back(aux[cur].total_dist);
        parr.push_back(hr);

        const uint32_t pr = prev_state[cur];
        if (pr == NO_PREV) break;
        cur = pr;
    }

    std::reverse(ppx.begin(), ppx.end());
    std::reverse(ppy.begin(), ppy.end());
    std::reverse(plvl.begin(), plvl.end());
    std::reverse(pstay.begin(), pstay.end());
    std::reverse(pdist.begin(), pdist.end());
    std::reverse(parr.begin(), parr.end());

    out.px = std::move(ppx);
    out.py = std::move(ppy);
    out.levels = std::move(plvl);
    out.stay = std::move(pstay);
    out.total_dist = std::move(pdist);
    out.arrays = std::move(parr);
    out.ok = true;

    materialize_loiter_loops(out, danger_grid.data(), validity_cache,
                             grid_w, grid_h, width, height,
                             step_size, num_levels, max_hours, flight_speed);
    return out;
}

struct RefinedPath {
    std::vector<double> xs, ys;
    std::vector<int>    levels;
    std::vector<double> stay;
    std::vector<double> total_dist;
    std::vector<int>    arrays;
    std::vector<uint8_t> frozen;
};

static RefinedPath refine_path_to_goal_with_constraints(
        const std::vector<double>& path_x, const std::vector<double>& path_y,
        const std::vector<int>& path_levels, const std::vector<double>& path_stay,
        const std::vector<double>& path_total, const std::vector<int>& path_arrays,
        const std::vector<uint8_t>& path_frozen,
        int arrays_count, double goal_x, double goal_y, int goal_level,
        double max_step = 20.0) {

    RefinedPath r;
    if (path_x.empty()) return r;

    auto fz = [&](size_t i) -> uint8_t {
        return i < path_frozen.size() ? path_frozen[i] : 0;
    };

    auto buggy_hour = [&](double dist) -> int {
        return (int)distance_to_hour_index(dist,
                                           cfg::FLIGHT_SPEED,
                                           cfg::HOURS_PER_ARRAY_SLICE,
                                           (double)arrays_count);
    };

    r.xs.push_back(path_x[0]);
    r.ys.push_back(path_y[0]);
    r.levels.push_back(path_levels[0]);
    r.stay.push_back(path_stay[0]);
    r.total_dist.push_back(path_total[0]);
    r.arrays.push_back(path_arrays[0]);
    r.frozen.push_back(fz(0));

    for (size_t i = 0; i + 1 < path_x.size(); ++i) {
        const double sx = path_x[i],  sy = path_y[i];
        const double ex = path_x[i + 1], ey = path_y[i + 1];
        const int start_lvl = path_levels[i], end_lvl = path_levels[i + 1];
        const double start_stay = path_stay[i], end_stay = path_stay[i + 1];
        const double start_dist = path_total[i], end_dist = path_total[i + 1];

        const double segment_length = hypot2(ex - sx, ey - sy);

        if (segment_length <= max_step) {
            r.xs.push_back(ex); r.ys.push_back(ey);
            r.levels.push_back(end_lvl);
            r.stay.push_back(end_stay);
            r.total_dist.push_back(end_dist);
            r.arrays.push_back(path_arrays[i + 1]);
            r.frozen.push_back(fz(i + 1));
        } else {
            const int num_steps = std::max(2, (int)py_trunc_i(segment_length / max_step) + 1);
            for (int step = 1; step < num_steps; ++step) {
                const double ratio = (double)step / num_steps;
                const double x = sx + ratio * (ex - sx);
                const double y = sy + ratio * (ey - sy);
                const double interpolated_dist = start_dist + ratio * (end_dist - start_dist);

                int    interpolated_lvl;
                double interpolated_stay;
                if (start_lvl == end_lvl) {
                    interpolated_lvl = start_lvl;
                    interpolated_stay = std::max(0.0, start_stay - ratio * segment_length);
                } else {
                    if (ratio < 0.5) {
                        interpolated_lvl = start_lvl;
                        interpolated_stay = std::max(0.0, start_stay - ratio * segment_length);
                    } else {
                        interpolated_lvl = end_lvl;
                        const int level_diff = std::abs(end_lvl - start_lvl);
                        interpolated_stay = std::max(
                            0.0, cfg::LEVEL_STAY_MULTIPLIER * level_diff -
                                 (ratio - 0.5) * segment_length);
                    }
                }

                r.xs.push_back(x); r.ys.push_back(y);
                r.levels.push_back(interpolated_lvl);
                r.stay.push_back(interpolated_stay);
                r.total_dist.push_back(interpolated_dist);
                r.arrays.push_back(buggy_hour(interpolated_dist));

                r.frozen.push_back(fz(i) && fz(i + 1) ? 1 : 0);
            }
            r.xs.push_back(ex); r.ys.push_back(ey);
            r.levels.push_back(end_lvl);
            r.stay.push_back(end_stay);
            r.total_dist.push_back(end_dist);
            r.arrays.push_back(path_arrays[i + 1]);
            r.frozen.push_back(fz(i + 1));
        }
    }

    const double last_x = r.xs.back(), last_y = r.ys.back();
    const double dist_to_goal = hypot2(goal_x - last_x, goal_y - last_y);

    if (dist_to_goal > max_step) {
        const int num_steps = std::max(2, (int)py_trunc_i(dist_to_goal / max_step) + 1);
        for (int step = 1; step < num_steps; ++step) {
            const double ratio = (double)step / num_steps;
            const double x = last_x + ratio * (goal_x - last_x);
            const double y = last_y + ratio * (goal_y - last_y);
            const double interpolated_dist = r.total_dist.back() + ratio * dist_to_goal;
            const int    interpolated_lvl  = r.levels.back();
            const double interpolated_stay = std::max(0.0, r.stay.back() - ratio * dist_to_goal);

            r.xs.push_back(x); r.ys.push_back(y);
            r.levels.push_back(interpolated_lvl);
            r.stay.push_back(interpolated_stay);
            r.total_dist.push_back(interpolated_dist);
            r.arrays.push_back(buggy_hour(interpolated_dist));
            r.frozen.push_back(0);
        }
    }

    r.frozen.push_back(0);
    r.xs.push_back(goal_x); r.ys.push_back(goal_y);
    r.levels.push_back(goal_level);
    const size_t k = r.xs.size();
    const double goal_dist = r.total_dist.back() +
                             hypot2(goal_x - r.xs[k - 2], goal_y - r.ys[k - 2]);
    r.total_dist.push_back(goal_dist);
    r.stay.push_back(0.0);
    r.arrays.push_back(buggy_hour(goal_dist));

    {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "Path refined: %zu -> %zu points",
                      path_x.size(), r.xs.size());
        log_info(buf);
    }
    return r;
}

static void check_stay_requirements(const std::vector<int>& levels,
                                    const std::vector<double>& stay) {
    log_info("Checking stay requirements...");
    int violations = 0;
    for (size_t i = 1; i < levels.size(); ++i) {
        if (levels[i] > levels[i - 1]) {
            const double required = cfg::LEVEL_STAY_MULTIPLIER * (levels[i] - levels[i - 1]);
            if (stay[i] != required) violations++;
        }
    }
    if (violations == 0) log_info("All stay requirements satisfied");
    else log_warn("Violations found: " + std::to_string(violations));
}

static void analyze_array_usage(const std::vector<int>& arr_idx, int total_arrays) {
    log_info("3D ARRAY USAGE ANALYSIS:");
    std::map<int, int> usage;
    for (int i = 0; i < total_arrays; ++i) usage[i] = 0;
    for (int v : arr_idx) { auto it = usage.find(v); if (it != usage.end()) it->second++; }
    const size_t total_points = arr_idx.size();
    int unique_hours = 0;
    for (auto& kv : usage) {
        double pct = total_points == 0 ? 0.0 : (double)kv.second / total_points * 100.0;
        char buf[160];
        std::snprintf(buf, sizeof(buf), "  • Array %d: %d points (%.1f%%)",
                      kv.first, kv.second, pct);
        log_info(buf);
        if (kv.second > 0) unique_hours++;
    }
    log_info("  • Hours used: " + std::to_string(unique_hours) + " out of " +
             std::to_string(total_arrays));
}

struct Weights { double safety, length, smoothness, level_change; };

static Weights calculate_fixed_weights(double path_length) {
    double safety = cfg::SAFETY_WEIGHT *
                    (1.0 + cfg::SAFETY_WEIGHT_COEFF * (path_length / cfg::BASE_PATH_LENGTH));
    safety = std::max(cfg::SAFETY_WEIGHT, std::min(safety, cfg::SAFETY_WEIGHT * 3.0));
    double length = cfg::LENGTH_WEIGHT *
                    (1.0 + cfg::LENGTH_WEIGHT_COEFF * (path_length / cfg::BASE_PATH_LENGTH));
    length = std::max(cfg::LENGTH_WEIGHT, std::min(length, cfg::LENGTH_WEIGHT * 2.0));
    return {safety, length, cfg::SMOOTHNESS_WEIGHT, cfg::LEVEL_CHANGE_PENALTY};
}

static double path_fitness(const double* __restrict xs, const double* __restrict ys,
                           const int* __restrict levels,
                           const double* __restrict total_dists, int n,
                           const GridBool& validity_cache, const Grid2D& danger_cache_2d,
                           const std::vector<Array3D>& arrays_3d_list,
                           const double* level_penalties,
                           double scale_x, double scale_y, int max_x3d, int max_y3d,
                           double w_length, double w_safety, double w_smooth,
                           double w_level_change, double flight_speed, int max_hours) {
    if (n < 2) return 1e18;

    const int h = validity_cache.height, w = validity_cache.width;

    double total_length = 0.0, total_danger = 0.0, total_curvature = 0.0;
    long total_level_changes = 0;

    for (int i = 0; i < n; ++i) {
        long xi = py_round_i(xs[i]), yi = py_round_i(ys[i]);
        if (xi < 0 || xi >= w || yi < 0 || yi >= h) return 1e18;
        if (!validity_cache.at((int)yi, (int)xi)) return 1e18;
    }

    for (int i = 0; i < n - 1; ++i) {
        if (!line_clear(xs[i], ys[i], xs[i + 1], ys[i + 1], validity_cache)) return 1e18;
        total_length += hypot2(xs[i + 1] - xs[i], ys[i + 1] - ys[i]);
    }

    for (int i = 0; i < n; ++i) {
        const double time_hours = total_dists[i] / flight_speed;
        int array_idx = (int)py_trunc_i(time_hours);
        if (array_idx < 0) array_idx = 0;
        if (array_idx >= max_hours) array_idx = max_hours - 1;
        int x3d, y3d;
        map_to_3d_coords(xs[i], ys[i], scale_x, scale_y, max_x3d, max_y3d, x3d, y3d);
        const int lvl = levels[i];
        const Array3D& arr = arrays_3d_list[array_idx];
        const double level_danger = (lvl < arr.levels) ? arr.at(lvl, y3d, x3d) : 0.0;
        const double base_danger =
            danger_cache_2d.at((int)py_round_i(ys[i]), (int)py_round_i(xs[i]));
        total_danger += base_danger + level_danger + level_penalties[lvl];
    }

    for (int i = 1; i < n - 1; ++i) {
        const double dx1 = xs[i] - xs[i - 1], dy1 = ys[i] - ys[i - 1];
        const double dx2 = xs[i + 1] - xs[i], dy2 = ys[i + 1] - ys[i];
        const double norm1 = hypot2(dx1, dy1), norm2 = hypot2(dx2, dy2);
        if (norm1 > 0.0 && norm2 > 0.0) {
            double cos_angle = (dx1 * dx2 + dy1 * dy2) / (norm1 * norm2);
            if (cos_angle > 1.0) cos_angle = 1.0;
            if (cos_angle < -1.0) cos_angle = -1.0;
            const double span = hypot2(xs[i + 1] - xs[i - 1], ys[i + 1] - ys[i - 1]);
            total_curvature += (1.0 - cos_angle) * span;
        }
    }

    for (int i = 1; i < n; ++i)
        if (levels[i] != levels[i - 1]) total_level_changes++;

    return w_length * total_length + w_safety * total_danger +
           w_smooth * total_curvature + w_level_change * (double)total_level_changes;
}

static int find_optimal_level_with_constraints(double px, double py, int current_level,
                                               double current_level_distance,
                                               double total_distance, Environment& env) {
    int best_level = current_level;
    std::pair<double, int> best =
        env.danger_cache->get_total_danger_cached(px, py, current_level, total_distance);

    for (int off = -cfg::LEVEL_OPTIMIZATION_RANGE; off <= cfg::LEVEL_OPTIMIZATION_RANGE; ++off) {
        const int cand = current_level + off;
        if (cand < 0 || cand >= cfg::NUM_LEVELS) continue;
        if (cand > current_level) {
            if (current_level_distance < cfg::LEVEL_STAY_MULTIPLIER * (cand - current_level))
                continue;
        }
        std::pair<double, int> d =
            env.danger_cache->get_total_danger_cached(px, py, cand, total_distance);

        if (d.first < best.first || (d.first == best.first && d.second < best.second)) {
            best = d;
            best_level = cand;
        }
    }
    return best_level;
}

static void generate_safe_candidates(const std::vector<double>& path_xs,
                                     const std::vector<double>& path_ys,
                                     int center_index, double cur_x, double cur_y,
                                     Environment& env, const uint8_t* frozen,
                                     std::vector<double>& out_x, std::vector<double>& out_y) {
    out_x.clear();
    out_y.clear();

    const int num_candidates = cfg::NUM_CANDIDATES;
    const int max_attempts = num_candidates * 10;

    bool use_direction = false;
    double dir_x = 0.0, dir_y = 0.0;
    if (center_index > 0 && center_index < (int)path_xs.size() - 1) {
        const double ideal_x = (path_xs[center_index - 1] + path_xs[center_index + 1]) / 2.0;
        const double ideal_y = (path_ys[center_index - 1] + path_ys[center_index + 1]) / 2.0;
        const double dx = ideal_x - cur_x, dy = ideal_y - cur_y;
        const double dist_to_ideal = std::sqrt(dx * dx + dy * dy);
        if (dist_to_ideal > 0.0) {
            dir_x = dx / dist_to_ideal;
            dir_y = dy / dist_to_ideal;
            use_direction = true;
        }
    }

    std::vector<double> cx, cy;
    generate_candidates(cur_x, cur_y, num_candidates, max_attempts, cfg::OPTIMIZATION_RADIUS,
                        (double)env.width, (double)env.height, use_direction, dir_x, dir_y,
                        cx, cy);

    for (size_t i = 0; i < cx.size(); ++i) {
        if (is_candidate_safe(path_xs, path_ys, center_index, cx[i], cy[i],
                              cfg::POINTS_TO_ADJUST, env.map2_cache->validity_cache,
                              frozen)) {
            out_x.push_back(cx[i]);
            out_y.push_back(cy[i]);
        }
    }
}

struct OptimizeResult {
    std::vector<double> xs, ys;
    std::vector<int>    levels;
    std::vector<double> stay_req;
    std::vector<double> total_dist;
    std::vector<int>    array_idx;
    std::vector<double> fitness_progress;
};

static OptimizeResult optimize_path_with_fixed_weights(
        const std::vector<double>& init_xs, const std::vector<double>& init_ys,
        const std::vector<int>& init_levels, const std::vector<double>& init_stay,
        const std::vector<double>& init_total, const std::vector<int>& init_arr,
        const std::vector<uint8_t>& frozen_pts, const Weights& w, Environment& env) {

    const uint8_t* frozen =
        frozen_pts.size() == init_xs.size() ? frozen_pts.data() : nullptr;
    {
        size_t nfz = 0;
        for (uint8_t f : frozen_pts) nfz += f;
        if (nfz > 0)
            log_info("Заморожено точек ожидания (стадия 2 их не трогает): " +
                     std::to_string(nfz));
    }

    log_info("Launching path optimization with fixed weights");
    {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
                      "   • Safety %.2f | Length %.2f | Smooth %.0f | LevelChange %.2f",
                      w.safety, w.length, w.smoothness, w.level_change);
        log_info(buf);
    }

    Timer opt_timer;

    std::vector<double> path_xs = init_xs, path_ys = init_ys;
    std::vector<int>    path_levels = init_levels;
    std::vector<double> path_stay_requirements = init_stay;
    std::vector<double> path_total_distances = init_total;
    std::vector<int>    path_array_indices = init_arr;

    const int n_points = (int)path_xs.size();
    std::vector<double> lvl_dist = level_distances(path_xs, path_ys, path_levels);
    std::vector<double> best_fitness_progress;

    auto fitness_raw = [&](const double* xs, const double* ys, const int* lv,
                           const double* td, int n) {
        return path_fitness(xs, ys, lv, td, n,
                            env.map2_cache->validity_cache, env.map2_cache->danger_cache,
                            env.arrays_3d, cfg::LEVEL_PENALTIES,
                            env.SCALE_X, env.SCALE_Y, env.MAX_X_3D, env.MAX_Y_3D,
                            w.length, w.safety, w.smoothness, w.level_change,
                            cfg::FLIGHT_SPEED, (int)env.arrays_3d.size());
    };
    auto fitness_of = [&](const std::vector<double>& xs, const std::vector<double>& ys,
                          const std::vector<int>& lv, const std::vector<double>& td) {
        return fitness_raw(xs.data(), ys.data(), lv.data(), td.data(), (int)xs.size());
    };

    const size_t MAXC = (size_t)cfg::NUM_CANDIDATES + 1;
    std::vector<double> cand_tx(MAXC * n_points), cand_ty(MAXC * n_points),
                        cand_ttd(MAXC * n_points), cand_fit(MAXC);
    std::vector<int>    cand_lv(MAXC * n_points);

    for (int iteration = 0; iteration < cfg::NUM_ITERATIONS; ++iteration) {
        Timer it_timer;
        {
            char buf[80];
            std::snprintf(buf, sizeof(buf), "  Iteration %d/%d",
                          iteration + 1, cfg::NUM_ITERATIONS);
            log_info(buf);
        }
        long iteration_improvement = 0;

        double best_fitness = fitness_of(path_xs, path_ys, path_levels, path_total_distances);

        for (int i = 1; i < n_points - 1; ++i) {
            if (frozen && frozen[i]) continue;

            const double cur_x = path_xs[i], cur_y = path_ys[i];
            const int    current_level = path_levels[i];
            const double current_level_distance = lvl_dist[i];

            double best_cx = cur_x, best_cy = cur_y;
            int    best_level = current_level;

            std::vector<double> cand_x, cand_y;
            generate_safe_candidates(path_xs, path_ys, i, cur_x, cur_y, env, frozen,
                                     cand_x, cand_y);
            if (is_point_and_neighbors_safe(path_xs, path_ys, i, cur_x, cur_y,
                                            env.map2_cache->validity_cache)) {
                cand_x.push_back(cur_x);
                cand_y.push_back(cur_y);
            }
            if (cand_x.empty()) continue;

            const size_t ncand = cand_x.size();

            {
                std::vector<double> tx, ty, sx, sy;
                for (size_t c = 0; c < ncand; ++c) {
                    tx = path_xs;
                    ty = path_ys;
                    tx[i] = cand_x[c];
                    ty[i] = cand_y[c];
                    smooth_adjust_neighbors(tx, ty, i, cfg::POINTS_TO_ADJUST,
                                            env.map2_cache->validity_cache, frozen,
                                            sx, sy);

                    double* __restrict dx = &cand_tx[c * n_points];
                    double* __restrict dy = &cand_ty[c * n_points];
                    double* __restrict dd = &cand_ttd[c * n_points];
                    int*    __restrict dl = &cand_lv[c * n_points];

                    std::copy(sx.begin(), sx.end(), dx);
                    std::copy(sy.begin(), sy.end(), dy);

                    dd[0] = 0.0;
                    for (int k = 1; k < n_points; ++k)
                        dd[k] = dd[k - 1] + hypot2(dx[k] - dx[k - 1], dy[k] - dy[k - 1]);

                    std::copy(path_levels.begin(), path_levels.end(), dl);
                    dl[i] = find_optimal_level_with_constraints(
                        cand_x[c], cand_y[c], current_level, current_level_distance,
                        dd[i], env);
                }
            }

            parallel_for(0, ncand, [&](size_t c) {
                cand_fit[c] = fitness_raw(&cand_tx[c * n_points], &cand_ty[c * n_points],
                                          &cand_lv[c * n_points], &cand_ttd[c * n_points],
                                          n_points);
            });

            for (size_t c = 0; c < ncand; ++c) {
                if (cand_fit[c] < best_fitness) {
                    best_fitness = cand_fit[c];
                    best_cx = cand_x[c];
                    best_cy = cand_y[c];
                    best_level = cand_lv[c * n_points + i];
                    iteration_improvement++;
                }
            }

            path_xs[i] = best_cx;
            path_ys[i] = best_cy;
            path_levels[i] = best_level;

            std::vector<double> sx, sy;
            smooth_adjust_neighbors(path_xs, path_ys, i, cfg::POINTS_TO_ADJUST,
                                    env.map2_cache->validity_cache, frozen, sx, sy);
            path_xs.swap(sx);
            path_ys.swap(sy);

            path_total_distances = recalculate_distances(path_xs, path_ys);
            path_array_indices.assign(path_total_distances.size(), 0);
            for (size_t k = 0; k < path_total_distances.size(); ++k)
                path_array_indices[k] = (int)distance_to_hour_index(
                    path_total_distances[k], (double)env.arrays_3d.size());
            lvl_dist = level_distances(path_xs, path_ys, path_levels);
        }

        path_stay_requirements = recompute_stay_requirements(path_levels);
        const double current_fitness =
            fitness_of(path_xs, path_ys, path_levels, path_total_distances);
        best_fitness_progress.push_back(current_fitness);

        char buf[200];
        std::snprintf(buf, sizeof(buf),
                      "  Improved points: %ld, Fitness: %.2f, Time: %.2fs",
                      iteration_improvement, current_fitness, it_timer.sec());
        log_info(buf);
    }

    {
        char buf[120];
        std::snprintf(buf, sizeof(buf), "Optimization finished in %.2f seconds", opt_timer.sec());
        log_info(buf);
    }

    OptimizeResult r;
    r.xs = std::move(path_xs);
    r.ys = std::move(path_ys);
    r.levels = std::move(path_levels);
    r.stay_req = std::move(path_stay_requirements);
    r.total_dist = std::move(path_total_distances);
    r.array_idx = std::move(path_array_indices);
    r.fitness_progress = std::move(best_fitness_progress);
    return r;
}

struct SegmentResult {
    std::vector<double> d_xs, d_ys;
    std::vector<int>    d_levels;
    std::vector<double> d_stay, d_total;
    std::vector<int>    d_arr;
    double d_length = 0.0;

    std::vector<double> o_xs, o_ys;
    std::vector<int>    o_levels;
    std::vector<double> o_stay, o_total;
    std::vector<int>    o_arr;
    double o_length = 0.0;

    std::vector<double> fitness_progress;
    bool ok = false;
};

static double compute_path_length(const std::vector<double>& xs, const std::vector<double>& ys) {
    double s = 0.0;
    for (size_t i = 0; i + 1 < xs.size(); ++i)
        s += hypot2(xs[i + 1] - xs[i], ys[i + 1] - ys[i]);
    return s;
}

static SegmentResult run_segment_pipeline(int segment_idx,
                                          double sx, double sy, double ex, double ey,
                                          Environment& env,
                                          int start_level_in, int end_level_in,
                                          int hour_lookahead, DijkstraArena& arena) {
    SegmentResult res;

    log_info("============================================================");
    {
        char buf[200];
        std::snprintf(buf, sizeof(buf), "Segment %d: (%.1f, %.1f) -> (%.1f, %.1f)",
                      segment_idx, sx, sy, ex, ey);
        log_info(buf);
    }
    log_info("============================================================");

    Timer seg_timer;

    log_info("Stage 1: Dijkstra algorithm with static weights");
    Timer dij_timer;

    int start_level = start_level_in;
    int goal_level  = end_level_in;
    if (start_level < 0) start_level = env.danger_cache->find_best_level_for_point(sx, sy, 0.0);
    if (goal_level  < 0) goal_level  = env.danger_cache->find_best_level_for_point(ex, ey, 0.0);

    {
        char buf[200];
        std::snprintf(buf, sizeof(buf), "   - Start: (%.1f, %.1f) level %d | Goal: (%.1f, %.1f) level %d",
                      sx, sy, start_level, ex, ey, goal_level);
        log_info(buf);
    }

    if (!env.map2_cache->get_cached_validity(sx, sy)) { log_error("Start point invalid"); return res; }
    if (!env.map2_cache->get_cached_validity(ex, ey)) { log_error("Goal point invalid");  return res; }

    DijkstraRaw raw = dijkstra_numba_grid(
        sx, sy, ex, ey, start_level, goal_level,
        env.map2_cache->validity_cache, env.map2_cache->danger_cache,
        env.arrays_3d, cfg::LEVEL_PENALTIES,
        cfg::DIJKSTRA_STEP_SIZE, cfg::LOOKAHEAD_LEVELS, cfg::LEVEL_STAY_MULTIPLIER,
        cfg::NUM_LEVELS, cfg::FLIGHT_SPEED, (int)env.arrays_3d.size(),
        cfg::SAFETY_WEIGHT * cfg::DIJKSTRA_DANGER_WEIGHT,
        cfg::LENGTH_WEIGHT * cfg::DIJKSTRA_LENGTH_PENALTY,
        cfg::DIJKSTRA_GOAL_TOLERANCE, cfg::MAX_NODES,
        env.SCALE_X, env.SCALE_Y, env.MAX_X_3D, env.MAX_Y_3D,
        hour_lookahead, cfg::HOUR_STAY_DISTANCE, cfg::HOUR_SWITCH_PENALTY, arena);

    if (!raw.ok || raw.px.empty()) {
        log_error("Numba kernel failed to build a path");
        return res;
    }
    log_info("Path found by numba kernel");

    RefinedPath ref = refine_path_to_goal_with_constraints(
        raw.px, raw.py, raw.levels, raw.stay, raw.total_dist, raw.arrays,
        raw.frozen, (int)env.arrays_3d.size(), ex, ey, goal_level);

    analyze_array_usage(ref.arrays, (int)env.arrays_3d.size());
    check_stay_requirements(ref.levels, ref.stay);

    const double dij_time = dij_timer.sec();

    res.d_xs = ref.xs; res.d_ys = ref.ys;
    res.d_levels = ref.levels;
    res.d_stay = ref.stay;
    res.d_total = ref.total_dist;
    res.d_arr = ref.arrays;
    res.d_length = compute_path_length(res.d_xs, res.d_ys);

    {
        int changes = 0;
        for (size_t i = 1; i < res.d_levels.size(); ++i)
            if (res.d_levels[i] != res.d_levels[i - 1]) changes++;
        char buf[240];
        std::snprintf(buf, sizeof(buf),
                      "Dijkstra: %zu points, length %.2f px, level changes %d, time %.2fs",
                      res.d_xs.size(), res.d_length, changes, dij_time);
        log_info(buf);
    }

    const Weights fixed_weights = calculate_fixed_weights(res.d_length);

    log_info("Stage 2: Path optimization with fixed weights");
    OptimizeResult opt = optimize_path_with_fixed_weights(
        res.d_xs, res.d_ys, res.d_levels, res.d_stay, res.d_total, res.d_arr,
        ref.frozen, fixed_weights, env);

    res.o_xs = std::move(opt.xs);
    res.o_ys = std::move(opt.ys);
    res.o_levels = std::move(opt.levels);
    res.o_stay = std::move(opt.stay_req);
    res.o_total = std::move(opt.total_dist);
    res.o_arr = std::move(opt.array_idx);
    res.fitness_progress = std::move(opt.fitness_progress);
    res.o_length = compute_path_length(res.o_xs, res.o_ys);

    {
        char buf[200];
        std::snprintf(buf, sizeof(buf), "Optimized: %zu points, length %.2f px",
                      res.o_xs.size(), res.o_length);
        log_info(buf);
        std::snprintf(buf, sizeof(buf), "Segment %d total time: %.2fs", segment_idx, seg_timer.sec());
        log_info(buf);
    }

    res.ok = true;
    return res;
}

struct CombinedResult {
    std::vector<double> d_xs, d_ys;
    std::vector<int>    d_levels;
    std::vector<double> o_xs, o_ys;
    std::vector<int>    o_levels;
    bool ok = false;
};

static CombinedResult stitch_segment_results(const std::vector<SegmentResult>& segs,
                                             Environment& ,
                                             const std::vector<int>* route_levels) {
    CombinedResult c;
    if (segs.empty()) return c;

    std::vector<int> waypoint_indices{0};
    long total_len = 0;

    for (size_t idx = 0; idx < segs.size(); ++idx) {
        const bool skip_first = idx > 0;
        const size_t start = skip_first ? 1 : 0;
        const SegmentResult& s = segs[idx];

        for (size_t k = start; k < s.d_xs.size(); ++k) {
            c.d_xs.push_back(s.d_xs[k]);
            c.d_ys.push_back(s.d_ys[k]);
        }
        for (size_t k = start; k < s.o_xs.size(); ++k) {
            c.o_xs.push_back(s.o_xs[k]);
            c.o_ys.push_back(s.o_ys[k]);
        }
        for (size_t k = start; k < s.d_levels.size(); ++k) c.d_levels.push_back(s.d_levels[k]);
        for (size_t k = start; k < s.o_levels.size(); ++k) c.o_levels.push_back(s.o_levels[k]);

        total_len += (long)(s.d_levels.size() - start);
        waypoint_indices.push_back((int)total_len - 1);
    }

    if (route_levels && route_levels->size() == waypoint_indices.size()) {
        for (size_t i = 0; i < waypoint_indices.size(); ++i) {
            const int wp = waypoint_indices[i];
            if (wp >= 0 && wp < (int)c.d_levels.size()) c.d_levels[wp] = (*route_levels)[i];
            if (wp >= 0 && wp < (int)c.o_levels.size()) c.o_levels[wp] = (*route_levels)[i];
        }
    }

    c.ok = true;
    return c;
}

struct RouteRow { long long y, x, level; };

static std::vector<RouteRow> build_route_response(const CombinedResult& cr) {
    std::vector<RouteRow> out;
    if (!cr.ok) return out;

    const std::vector<double>* px = &cr.o_xs;
    const std::vector<double>* py = &cr.o_ys;
    const std::vector<int>*    lv = &cr.o_levels;
    if (px->empty()) { px = &cr.d_xs; py = &cr.d_ys; lv = &cr.d_levels; }
    if (px->empty()) return out;
    if (lv->empty()) lv = &cr.d_levels;

    for (size_t i = 0; i < px->size(); ++i) {
        RouteRow r;
        const long long xi = (long long)py_round_i((*px)[i]);
        const long long yi = (long long)py_round_i((*py)[i]);
        const long long li = (i < lv->size()) ? (long long)(*lv)[i] : 0LL;

        r.y = yi;
        r.x = xi;
        r.level = li;
        out.push_back(r);
    }
    return out;
}

static uint32_t crc_table[256];
static bool crc_table_ready = false;

static void make_crc_table() {
    for (uint32_t n = 0; n < 256; ++n) {
        uint32_t c = n;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        crc_table[n] = c;
    }
    crc_table_ready = true;
}

static uint32_t crc32_buf(const uint8_t* buf, size_t len, uint32_t crc = 0xFFFFFFFFu) {
    if (!crc_table_ready) make_crc_table();
    for (size_t n = 0; n < len; ++n) crc = crc_table[(crc ^ buf[n]) & 0xFF] ^ (crc >> 8);
    return crc;
}

static uint32_t adler32_buf(const uint8_t* data, size_t len) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < len; ++i) {
        a = (a + data[i]) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

static void put_u32be(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)(x >> 24));
    v.push_back((uint8_t)(x >> 16));
    v.push_back((uint8_t)(x >> 8));
    v.push_back((uint8_t)x);
}

static void write_chunk(std::ofstream& f, const char type[4], const std::vector<uint8_t>& data) {
    std::vector<uint8_t> hdr;
    put_u32be(hdr, (uint32_t)data.size());
    f.write((const char*)hdr.data(), hdr.size());

    std::vector<uint8_t> tc(type, type + 4);
    f.write((const char*)tc.data(), 4);
    if (!data.empty()) f.write((const char*)data.data(), data.size());

    uint32_t crc = crc32_buf(tc.data(), 4);
    if (!data.empty()) crc = crc32_buf(data.data(), data.size(), crc);
    crc ^= 0xFFFFFFFFu;

    std::vector<uint8_t> ce;
    put_u32be(ce, crc);
    f.write((const char*)ce.data(), ce.size());
}

static bool write_png(const std::string& path, int width, int height,
                      const std::vector<uint8_t>& rgb) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    f.write((const char*)sig, 8);

    std::vector<uint8_t> ihdr;
    put_u32be(ihdr, (uint32_t)width);
    put_u32be(ihdr, (uint32_t)height);
    ihdr.push_back(8);
    ihdr.push_back(2);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    write_chunk(f, "IHDR", ihdr);

    std::vector<uint8_t> raw;
    raw.reserve((size_t)height * (1 + (size_t)width * 3));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        const uint8_t* row = &rgb[(size_t)y * width * 3];
        raw.insert(raw.end(), row, row + (size_t)width * 3);
    }

    std::vector<uint8_t> z;
    z.push_back(0x78);
    z.push_back(0x01);
    size_t pos = 0;
    while (pos < raw.size()) {
        const size_t block = std::min<size_t>(65535, raw.size() - pos);
        const bool final_block = (pos + block >= raw.size());
        z.push_back(final_block ? 1 : 0);
        z.push_back((uint8_t)(block & 0xFF));
        z.push_back((uint8_t)((block >> 8) & 0xFF));
        const uint16_t nlen = (uint16_t)(~(uint16_t)block);
        z.push_back((uint8_t)(nlen & 0xFF));
        z.push_back((uint8_t)((nlen >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + block);
        pos += block;
    }
    put_u32be(z, adler32_buf(raw.data(), raw.size()));

    write_chunk(f, "IDAT", z);
    write_chunk(f, "IEND", {});
    return true;
}

struct Canvas {
    int w = 0, h = 0;
    std::vector<uint8_t> px;
    void init(int W, int H) { w = W; h = H; px.assign((size_t)W * H * 3, 255); }
    inline void set(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        size_t i = ((size_t)y * w + x) * 3;
        px[i] = r; px[i + 1] = g; px[i + 2] = b;
    }
    void disc(double cx, double cy, double rad, uint8_t r, uint8_t g, uint8_t b) {
        const int x0 = (int)std::floor(cx - rad), x1 = (int)std::ceil(cx + rad);
        const int y0 = (int)std::floor(cy - rad), y1 = (int)std::ceil(cy + rad);
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const double dx = x - cx, dy = y - cy;
                if (dx * dx + dy * dy <= rad * rad) set(x, y, r, g, b);
            }
    }
    void line(double x0, double y0, double x1, double y1, double thick,
              uint8_t r, uint8_t g, uint8_t b) {
        const double dx = x1 - x0, dy = y1 - y0;
        const double len = std::sqrt(dx * dx + dy * dy);
        const int steps = std::max(1, (int)std::ceil(len));
        for (int s = 0; s <= steps; ++s) {
            const double t = (double)s / steps;
            disc(x0 + dx * t, y0 + dy * t, thick, r, g, b);
        }
    }

    void star(double cx, double cy, double R, uint8_t r, uint8_t g, uint8_t b) {
        const int N = 10;
        double vx[N], vy[N];
        const double inner = R * 0.382;
        for (int i = 0; i < N; ++i) {
            const double rad = (i % 2 == 0) ? R : inner;
            const double a = -M_PI / 2.0 + i * M_PI / 5.0;
            vx[i] = cx + rad * std::cos(a);
            vy[i] = cy + rad * std::sin(a);
        }
        const int x0 = (int)std::floor(cx - R), x1 = (int)std::ceil(cx + R);
        const int y0 = (int)std::floor(cy - R), y1 = (int)std::ceil(cy + R);
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                bool inside = false;
                for (int i = 0, j = N - 1; i < N; j = i++) {
                    if (((vy[i] > y) != (vy[j] > y)) &&
                        (x < (vx[j] - vx[i]) * (y - vy[i]) / (vy[j] - vy[i]) + vx[i]))
                        inside = !inside;
                }
                if (inside) set(x, y, r, g, b);
            }
        }
    }
};

static void visualize(const Grid2D& danger_map, const std::vector<RouteRow>& route,
                      double start_x, double start_y, double goal_x, double goal_y,
                      const std::string& save_path) {

    const int MAXDIM = 1800;
    int factor = 1;
    while (danger_map.width / factor > MAXDIM || danger_map.height / factor > MAXDIM) factor++;

    const int W = std::max(1, danger_map.width / factor);
    const int H = std::max(1, danger_map.height / factor);

    Canvas cv;
    cv.init(W, H);

    double vmin = danger_map.v.empty() ? 0.0 : danger_map.v[0];
    double vmax = vmin;
    for (double v : danger_map.v) { vmin = std::min(vmin, v); vmax = std::max(vmax, v); }
    const double span = (vmax > vmin) ? (vmax - vmin) : 1.0;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int sy = std::min(danger_map.height - 1, y * factor);
            const int sx = std::min(danger_map.width - 1, x * factor);
            const double norm = (danger_map.at(sy, sx) - vmin) / span;
            const double gray = 1.0 - norm;
            const double shown = 0.8 * gray + 0.2 * 1.0;
            const uint8_t c = (uint8_t)std::lround(std::max(0.0, std::min(1.0, shown)) * 255.0);
            cv.set(x, y, c, c, c);
        }
    }

    const double sc = 1.0 / factor;
    const double lw = std::max(1.0, 1.2);

    for (size_t i = 0; i + 1 < route.size(); ++i) {
        cv.line((double)route[i].x * sc,     (double)route[i].y * sc,
                (double)route[i + 1].x * sc, (double)route[i + 1].y * sc,
                lw, 214, 39, 40);
    }

    cv.star(start_x * sc, start_y * sc, 9.0, 0, 150, 0);
    cv.star(goal_x  * sc, goal_y  * sc, 9.0, 0, 60, 220);

    if (write_png(save_path, W, H, cv.px))
        std::cout << "Карта сохранена: " << save_path << " (" << W << "x" << H << ")\n";
    else
        log_error("Не удалось записать " + save_path);
}

static std::vector<RouteRow> calculate_path(const std::vector<std::pair<double, double>>& route_yx,
                                            const std::string& map_file,
                                            const std::vector<std::string>& array_files,
                                            Environment*& env_out) {

    std::vector<std::pair<double, double>> route_xy;
    for (auto& p : route_yx) route_xy.emplace_back(p.second, p.first);

    Environment* env = new Environment(map_file, array_files);
    env_out = env;

    log_info("============================================================");
    log_info("Full algorithm: Dijkstra (static) + Optimization (fixed)");
    log_info("============================================================");

    if (route_xy.size() < 2) {
        log_error("At least two route points are required");
        return {};
    }

    Timer total_timer;

    log_info("Checking all route points...");
    for (size_t i = 0; i < route_xy.size(); ++i) {
        const double x = route_xy[i].first, y = route_xy[i].second;
        if (!(x >= 0 && x < env->width && y >= 0 && y < env->height)) {
            char buf[220];
            std::snprintf(buf, sizeof(buf),
                          "ValueError: Route point %zu (%.1f, %.1f) is out of bounds for map %dx%d",
                          i + 1, x, y, env->width, env->height);
            throw std::runtime_error(buf);
        }
    }
    log_info("All route points are valid");
    log_info("Total segments: " + std::to_string(route_xy.size() - 1));

    const int hour_lookahead = get_hour_lookahead_value((int)env->arrays_3d.size());

    DijkstraArena arena;

    std::vector<SegmentResult> segs;
    for (size_t i = 0; i + 1 < route_xy.size(); ++i) {
        SegmentResult sr = run_segment_pipeline((int)i + 1,
                                                route_xy[i].first, route_xy[i].second,
                                                route_xy[i + 1].first, route_xy[i + 1].second,
                                                *env, -1, -1, hour_lookahead, arena);
        if (!sr.ok) {
            log_error("Segment " + std::to_string(i + 1) + " failed, route not built");
            return {};
        }
        segs.push_back(std::move(sr));
    }

    CombinedResult cr = stitch_segment_results(segs, *env, nullptr);
    if (!cr.ok) { log_error("Failed to stitch segment results"); return {}; }

    {
        char buf[120];
        std::snprintf(buf, sizeof(buf), "Total execution time: %.2fs", total_timer.sec());
        log_info(buf);
    }

    return build_route_response(cr);
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    g_rng.seed(cfg::RNG_SEED);

    Timer wall;

    std::vector<std::pair<double, double>> route_points = {
        {2000.0, 8000.0},
        {3000.0, 3850.0},
        {1500.0, 1500.0},
    };

    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--threads=", 0) == 0) g_threads = (unsigned)std::atoi(a.c_str() + 10);
        else if (a.rfind("--wastar=", 0) == 0) g_wastar = std::atof(a.c_str() + 9);
        else pos.push_back(a);
    }
    if (!(g_wastar >= 1.0)) g_wastar = 1.0;
    if (g_wastar != 1.0) {
        char buf[120];
        std::snprintf(buf, sizeof(buf), "Взвешенный A*: W=%.2f", g_wastar);
        log_info(buf);
    }
#ifdef _OPENMP
    if (g_threads) omp_set_num_threads((int)g_threads);
#endif
    if (pos.size() >= 4 && pos.size() % 2 == 0) {
        route_points.clear();
        for (size_t i = 0; i + 1 < pos.size(); i += 2)
            route_points.emplace_back(std::atof(pos[i].c_str()), std::atof(pos[i + 1].c_str()));
    }
    {
        char buf[120];
        std::snprintf(buf, sizeof(buf), "Потоков: %u", thread_count());
        log_info(buf);
    }

    const std::string data_dir = "data";
    const std::string map_file = data_dir + "/map-test.npy";
    std::vector<std::string> array_files;
    for (int h = 0; h < 10; ++h)
        array_files.push_back(data_dir + "/" + std::to_string(h) + "h.npy");

    Environment* env = nullptr;
    std::vector<RouteRow> result;

    try {
        result = calculate_path(route_points, map_file, array_files, env);
    } catch (const std::exception& e) {
        log_error(e.what());
        delete env;
        std::cout << "\nВремя выполнения: " << wall.sec() << " c\n";
        return 1;
    }

    if (result.empty()) {
        std::cout << "Маршрут не построен\n";
        delete env;
        std::cout << "\nВремя выполнения: " << wall.sec() << " c\n";
        return 1;
    }

    const double compute_time = wall.sec();

    std::cout << "\nМаршрут (" << result.size() << " точек), формат (y, x, level):\n[";
    for (size_t i = 0; i < result.size(); ++i) {
        if (i) std::cout << "\n ";
        std::cout << "[" << result[i].y << " " << result[i].x << " " << result[i].level << "]";
    }
    std::cout << "]\n\n";

    const double start_x = route_points.front().second;
    const double start_y = route_points.front().first;
    const double goal_x  = route_points.back().second;
    const double goal_y  = route_points.back().first;

    Timer vis_timer;
    visualize(env->raw_map, result, start_x, start_y, goal_x, goal_y, "cpp_test.png");
    const double visualize_time = vis_timer.sec();

    delete env;

    std::cout << "\n=============================================\n";
    std::cout << "Время выполнения: " << compute_time << " c\n";
    std::cout << "  (визуализация, не входит в замер: " << visualize_time << " c)\n";
    std::cout << "=============================================\n";
    return 0;
}
