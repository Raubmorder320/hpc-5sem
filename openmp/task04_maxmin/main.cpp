// OpenMP-4. Максимум среди минимальных элементов строк матрицы:
//   y = max_i min_j a_ij
// n из --sizes — размер матрицы N (N x N).
#include "bench.hpp"

#include <cmath>

double maxmin_seq(const std::vector<double>& a, long n) {
    double res = -INFINITY;
    for (long i = 0; i < n; ++i) {
        const double* row = &a[i * n];
        double rmin = row[0];
        for (long j = 1; j < n; ++j)
            if (row[j] < rmin) rmin = row[j];
        if (rmin > res) res = rmin;
    }
    return res;
}

// Параллелится внешний цикл: каждый поток обрабатывает свои строки целиком.
// rmin объявлена внутри тела цикла => приватна; res собирается через reduction(max).
// Внутренний цикл последовательный: строка лежит в памяти подряд, поток читает её линейно.
double maxmin_par(const std::vector<double>& a, long n, int threads) {
    const double* p = a.data();
    double res = -INFINITY;
#pragma omp parallel for reduction(max : res) num_threads(threads)
    for (long i = 0; i < n; ++i) {
        const double* row = p + i * n;
        double rmin = row[0];
        for (long j = 1; j < n; ++j)
            if (row[j] < rmin) rmin = row[j];
        if (rmin > res) res = rmin;
    }
    return res;
}

int main(int argc, char** argv) {
    auto args = bench::parse_args(argc, argv, {100, 1'000, 5'000, 10'000});
    bench::csv_header();

    for (long n : args.sizes) {
        auto a = bench::random_matrix(n);
        bench::check("maxmin", maxmin_par(a, n, 4), maxmin_seq(a, n));

        volatile double sink;
        bench::run_series("04", "outer_for", n, args, [&](int t) { sink = maxmin_par(a, n, t); });
        (void)sink;
    }
}
