// OpenMP-9. Вложенный параллелизм на примере задачи 4 (max по строкам от min по строке).
// Сравниваются: только внешний цикл, только внутренний, вложенные parallel for
// с разным делением потоков между уровнями.
// n из --sizes — размер матрицы N; --threads — общее число потоков.
#include "bench.hpp"

#include <algorithm>
#include <cmath>

double maxmin_seq(const std::vector<double>& a, long n) {
    double res = -INFINITY;
    for (long i = 0; i < n; ++i) {
        const double* row = &a[i * n];
        double rmin = row[0];
        for (long j = 1; j < n; ++j) rmin = std::min(rmin, row[j]);
        res = std::max(res, rmin);
    }
    return res;
}

// Уровень 1: строки делятся между потоками, одна параллельная область на весь расчёт.
double maxmin_outer(const std::vector<double>& a, long n, int threads) {
    const double* p = a.data();
    double res = -INFINITY;
#pragma omp parallel for reduction(max : res) num_threads(threads)
    for (long i = 0; i < n; ++i) {
        const double* row = p + i * n;
        double rmin = row[0];
        for (long j = 1; j < n; ++j) rmin = std::min(rmin, row[j]);
        res = std::max(res, rmin);
    }
    return res;
}

// Уровень 2: строки идут последовательно, параллелится поиск минимума внутри строки.
// Параллельная область открывается N раз — на каждую строку, отсюда большие накладные расходы.
double maxmin_inner(const std::vector<double>& a, long n, int threads) {
    const double* p = a.data();
    double res = -INFINITY;
    for (long i = 0; i < n; ++i) {
        const double* row = p + i * n;
        double rmin = INFINITY;
#pragma omp parallel for reduction(min : rmin) num_threads(threads)
        for (long j = 0; j < n; ++j) rmin = std::min(rmin, row[j]);
        res = std::max(res, rmin);
    }
    return res;
}

// Оба уровня: outer потоков делят строки, каждый из них для своей строки
// открывает вложенную команду из inner потоков. Всего потоков outer * inner.
// Без omp_set_nested(1) (или OMP_NESTED=true) внутренний parallel выполнялся бы одним потоком.
double maxmin_nested(const std::vector<double>& a, long n, int outer, int inner) {
    omp_set_nested(1);
    const double* p = a.data();
    double res = -INFINITY;
#pragma omp parallel for reduction(max : res) num_threads(outer)
    for (long i = 0; i < n; ++i) {
        const double* row = p + i * n;
        double rmin = INFINITY;
#pragma omp parallel for reduction(min : rmin) num_threads(inner)
        for (long j = 0; j < n; ++j) rmin = std::min(rmin, row[j]);
        res = std::max(res, rmin);
    }
    return res;
}

int main(int argc, char** argv) {
    auto args = bench::parse_args(argc, argv, {100, 1'000, 5'000});
    bench::csv_header();
    // Проверка поддержки вложенности (п. 5.7.7): если реализация её не поддерживает,
    // omp_set_nested(1) игнорируется и omp_get_nested() вернёт 0.
    omp_set_nested(1);
    std::fprintf(stderr, "omp_get_nested() = %d, max active levels supported: %d\n",
                 omp_get_nested(), omp_get_supported_active_levels());

    for (long n : args.sizes) {
        auto a = bench::random_matrix(n);
        double ref = maxmin_seq(a, n);
        bench::check("outer", maxmin_outer(a, n, 4), ref);
        bench::check("inner", maxmin_inner(a, n, 4), ref);
        if (n <= 1000) bench::check("nested", maxmin_nested(a, n, 2, 2), ref);

        volatile double sink;
        bench::run_seq("09", "sequential", n, args, [&] { sink = maxmin_seq(a, n); });
        bench::run_series("09", "outer", n, args, [&](int t) { sink = maxmin_outer(a, n, t); });
        bench::run_series("09", "inner", n, args, [&](int t) { sink = maxmin_inner(a, n, t); });

        // Ограничиваем число запусков вложенных команд на больших сериях.
        if (n > 1000) continue;
        bench::Args nested_args = args;
        std::erase_if(nested_args.threads, [](int t) { return t > 16; });
        // t потоков делятся между уровнями: 2 внешних × t/2 внутренних и наоборот
        bench::run_series("09", "nested_2xk", n, nested_args, [&](int t) {
            sink = maxmin_nested(a, n, std::min(2, t), std::max(1, t / 2));
        });
        bench::run_series("09", "nested_kx2", n, nested_args, [&](int t) {
            sink = maxmin_nested(a, n, std::max(1, t / 2), std::min(2, t));
        });
    }
}
