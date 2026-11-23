// OpenMP-1. Минимум (максимум) среди элементов вектора.
// Варианты: с reduction(min/max) и без редукции (локальный минимум потока + critical).
#include "bench.hpp"

#include <cmath>

double min_seq(const std::vector<double>& a) {
    double m = INFINITY;
    for (double x : a)
        if (x < m) m = x;
    return m;
}

// reduction(min:m): каждый поток получает свою копию m, инициализированную нейтральным
// элементом (+inf для min), считает минимум своей части итераций, а в конце региона
// OpenMP сам объединяет копии операцией min и записывает результат в исходную m.
double min_reduction(const std::vector<double>& a, int threads) {
    const long n = static_cast<long>(a.size());
    const double* p = a.data();
    double m = INFINITY;
#pragma omp parallel for reduction(min : m) num_threads(threads)
    for (long i = 0; i < n; ++i)
        if (p[i] < m) m = p[i];
    return m;
}

// То же вручную: локальная переменная потока + слияние в критической секции.
// Каждый поток объединяет локальный минимум с общим один раз.
double min_no_reduction(const std::vector<double>& a, int threads) {
    const long n = static_cast<long>(a.size());
    const double* p = a.data();
    double gmin = INFINITY;
#pragma omp parallel shared(gmin) num_threads(threads)
    {
        double lmin = INFINITY;  // объявлена внутри региона => приватная для потока
#pragma omp for nowait  // nowait: не ждать остальных после цикла, сразу идти сливать результат
        for (long i = 0; i < n; ++i)
            if (p[i] < lmin) lmin = p[i];
#pragma omp critical
        {
            if (lmin < gmin) gmin = lmin;
        }
    }  // неявный барьер в конце parallel: после него gmin окончателен
    return gmin;
}

int main(int argc, char** argv) {
    auto args = bench::parse_args(argc, argv, {1'000, 100'000, 10'000'000, 100'000'000});
    bench::csv_header();
    // _OPENMP — дата стандарта ГГГГММ; reduction(min/max) появилась в OpenMP 3.1 (201107),
    // в стандарте 2.5 из лекции её нет — тогда остаётся только вариант без редукции.
    std::fprintf(stderr, "_OPENMP = %d, reduction(min/max): %s\n", _OPENMP,
                 _OPENMP >= 201107 ? "есть" : "нет");

    for (long n : args.sizes) {
        auto a = bench::random_vector(n);
        double ref = min_seq(a);
        bench::check("reduction", min_reduction(a, 4), ref);
        bench::check("no_reduction", min_no_reduction(a, 4), ref);

        volatile double sink;
        bench::run_series("01", "reduction", n, args, [&](int t) { sink = min_reduction(a, t); });
        bench::run_series("01", "no_reduction", n, args, [&](int t) { sink = min_no_reduction(a, t); });
        (void)sink;
    }
}
