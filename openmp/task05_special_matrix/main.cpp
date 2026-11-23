// OpenMP-5. Задача 4 для матриц специального вида (треугольная, ленточная)
// при разных правилах распределения итераций (static / dynamic / guided).
// n из --sizes — размер матрицы N.
#include "bench.hpp"

#include <algorithm>
#include <cmath>
#include <string>

enum class MatrixKind { LowerTriangular, Band };

// Значимая часть строки i: [row_begin, row_end).
// Треугольная: j ∈ [0, i] — длина строки растёт от 1 до N, нагрузка итераций неравномерна.
// Ленточная: |i - j| <= band — длина почти у всех строк 2*band+1, нагрузка равномерна.
long row_begin(MatrixKind kind, long i, long n, long band) {
    if (kind == MatrixKind::LowerTriangular) return 0;
    return std::max(0L, i - band);
}

long row_end(MatrixKind kind, long i, long n, long band) {
    if (kind == MatrixKind::LowerTriangular) return i + 1;
    return std::min(n, i + band + 1);
}

double maxmin_special_seq(const std::vector<double>& a, long n, MatrixKind kind, long band) {
    double res = -INFINITY;
    for (long i = 0; i < n; ++i) {
        const double* row = &a[i * n];
        long lo = row_begin(kind, i, n, band), hi = row_end(kind, i, n, band);
        double rmin = row[lo];
        for (long j = lo + 1; j < hi; ++j)
            if (row[j] < rmin) rmin = row[j];
        res = std::max(res, rmin);
    }
    return res;
}

// schedule(runtime) берёт стратегию из внутренней переменной OpenMP (run-sched-var),
// которую задаёт omp_set_schedule. Так одна функция проверяет все стратегии.
// chunk = 0 означает «размер по умолчанию»: для static это N/threads подряд идущих строк.
double maxmin_special_par(const std::vector<double>& a, long n, MatrixKind kind, long band,
                          omp_sched_t sched, int chunk, int threads) {
    omp_set_schedule(sched, chunk);
    const double* p = a.data();
    double res = -INFINITY;
#pragma omp parallel for schedule(runtime) reduction(max : res) num_threads(threads)
    for (long i = 0; i < n; ++i) {
        const double* row = p + i * n;
        long lo = row_begin(kind, i, n, band), hi = row_end(kind, i, n, band);
        double rmin = row[lo];
        for (long j = lo + 1; j < hi; ++j)
            if (row[j] < rmin) rmin = row[j];
        if (rmin > res) res = rmin;
    }
    return res;
}

int main(int argc, char** argv) {
    // 1000–2000: матрица почти помещается в кэш L3, упор в вычисления — разница стратегий видна.
    // 5000+: упор в пропускную способность памяти, разница стратегий смазывается.
    auto args = bench::parse_args(argc, argv, {1'000, 2'000, 5'000, 10'000});
    bench::csv_header();

    struct Sched { const char* name; omp_sched_t kind; int chunk; };
    const Sched schedules[] = {
        {"static", omp_sched_static, 0},     // 0 = размер по умолчанию (n / threads)
        {"static_1", omp_sched_static, 1},
        {"dynamic_1", omp_sched_dynamic, 1},
        {"dynamic_16", omp_sched_dynamic, 16},
        {"guided_1", omp_sched_guided, 1},
    };
    struct Kind { const char* name; MatrixKind kind; };
    const Kind kinds[] = {{"triangular", MatrixKind::LowerTriangular}, {"band", MatrixKind::Band}};

    for (long n : args.sizes) {
        auto a = bench::random_matrix(n);
        const long band = n / 10;
        for (const auto& k : kinds) {
            double ref = maxmin_special_seq(a, n, k.kind, band);
            std::string seq_name = std::string(k.name) + "_sequential";
            volatile double seq_sink;
            bench::run_seq("05", seq_name.c_str(), n, args,
                           [&] { seq_sink = maxmin_special_seq(a, n, k.kind, band); });
            (void)seq_sink;
            for (const auto& s : schedules) {
                bench::check(s.name, maxmin_special_par(a, n, k.kind, band, s.kind, s.chunk, 4), ref);
                std::string variant = std::string(k.name) + "_" + s.name;
                volatile double sink;
                bench::run_series("05", variant.c_str(), n, args, [&](int t) {
                    sink = maxmin_special_par(a, n, k.kind, band, s.kind, s.chunk, t);
                });
                (void)sink;
            }
        }
    }
}
