// Общие утилиты для всех задач: разбор аргументов, генерация данных, замер времени, вывод CSV.
#pragma once

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace bench {

// ---------- аргументы командной строки ----------
// Пример: ./bin/task01 --sizes 1000,1000000 --threads 1,2,4,8 --reps 5
struct Args {
    std::vector<long> sizes;
    std::vector<int> threads;
    int reps = 5;
};

inline std::vector<long> parse_list(const char* s) {
    std::vector<long> out;
    std::string cur;
    for (const char* p = s;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!cur.empty()) out.push_back(std::stol(cur));
            cur.clear();
            if (*p == '\0') break;
        } else {
            cur += *p;
        }
    }
    return out;
}

// По умолчанию: 1, 2, 4, ... до числа ядер P, затем 2P — чтобы были случаи «меньше, равно, больше».
inline std::vector<int> default_threads() {
    int p = omp_get_num_procs();
    std::vector<int> t;
    for (int k = 1; k < p; k *= 2) t.push_back(k);
    t.push_back(p);
    t.push_back(2 * p);
    return t;
}

inline Args parse_args(int argc, char** argv, std::vector<long> default_sizes) {
    if (argc % 2 == 0) {
        std::fprintf(stderr, "для каждого параметра нужно значение\n");
        std::exit(1);
    }
    Args a;
    a.sizes = std::move(default_sizes);
    a.threads = default_threads();
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string key = argv[i];
        if (key == "--sizes") {
            a.sizes = parse_list(argv[i + 1]);
        } else if (key == "--threads") {
            a.threads.clear();
            for (long t : parse_list(argv[i + 1])) {
                if (t <= 0 || t > std::numeric_limits<int>::max()) std::exit(1);
                a.threads.push_back(static_cast<int>(t));
            }
        } else if (key == "--reps") {
            a.reps = std::atoi(argv[i + 1]);
        } else {
            std::fprintf(stderr, "неизвестный аргумент: %s\n", argv[i]);
            std::exit(1);
        }
    }
    if (a.reps <= 0 || a.sizes.empty() || a.threads.empty() ||
        std::any_of(a.sizes.begin(), a.sizes.end(), [](long n) { return n <= 0; })) {
        std::fprintf(stderr, "размеры, число потоков и число повторов должны быть положительными\n");
        std::exit(1);
    }
    return a;
}

// ---------- генерация данных ----------
inline std::vector<double> random_vector(long n, unsigned seed = 42, double lo = -1000.0, double hi = 1000.0) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(lo, hi);
    std::vector<double> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// Матрица N x N в одном массиве, доступ a[i * n + j].
inline std::vector<double> random_matrix(long n, unsigned seed = 42) {
    if (n <= 0 || n > std::numeric_limits<long>::max() / n) {
        std::fprintf(stderr, "неверный размер матрицы\n");
        std::exit(1);
    }
    return random_vector(n * n, seed);
}

// ---------- замер времени ----------
// Один прогрев, затем reps запусков; возвращает медиану в секундах.
inline double measure(const std::function<void()>& fn, int reps) {
    fn();
    std::vector<double> t(reps);
    for (int r = 0; r < reps; ++r) {
        double t0 = omp_get_wtime();
        fn();
        t[r] = omp_get_wtime() - t0;
    }
    std::sort(t.begin(), t.end());
    return t[reps / 2];
}

// ---------- вывод ----------
// CSV в stdout: task,variant,n,threads,time_s,speedup
// Ускорение считается относительно запуска на 1 потоке в той же серии.
inline void csv_header() {
    std::printf("task,variant,n,threads,time_s,speedup\n");
}

// Прогоняет fn(threads) для каждого числа потоков и печатает строки CSV.
// fn сам должен использовать переданное число потоков (omp_set_num_threads или num_threads()).
inline void run_series(const char* task, const char* variant, long n, const Args& a,
                       const std::function<void(int)>& fn) {
    double t1 = measure([&] { fn(1); }, a.reps);
    for (int th : a.threads) {
        double t = (th == 1) ? t1 : measure([&] { fn(th); }, a.reps);
        std::printf("%s,%s,%ld,%d,%.6f,%.3f\n", task, variant, n, th, t, t1 / t);
        std::fflush(stdout);
    }
}

// Последовательная версия (без OpenMP) — общая база, относительно которой plot.py считает
// ускорение всех вариантов задачи. Печатается одной строкой с threads = 1.
inline void run_seq(const char* task, const char* variant, long n, const Args& a,
                    const std::function<void()>& fn) {
    double t = measure(fn, a.reps);
    std::printf("%s,%s,%ld,1,%.6f,1.000\n", task, variant, n, t);
    std::fflush(stdout);
}

// Сравнение результата с эталоном (для double — с относительной погрешностью).
inline void check(const char* what, double got, double expected, double rel_eps = 1e-9) {
    double diff = std::abs(got - expected);
    double scale = std::max(1.0, std::abs(expected));
    if (got != expected && (!std::isfinite(diff) || diff > rel_eps * scale)) {
        std::fprintf(stderr, "[FAIL] %s: got %.12g, expected %.12g\n", what, got, expected);
        std::exit(1);
    }
}

}  // namespace bench
