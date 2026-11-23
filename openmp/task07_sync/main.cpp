// OpenMP-7. Редукция (сумма элементов) через разные способы взаимоисключения:
// atomic, critical, omp_lock_t — и сравнение с reduction(+:s).
// Для atomic/critical/lock — два варианта: синхронизация на каждой итерации
// и локальная частичная сумма + одна синхронизация на поток.
#include "bench.hpp"

double sum_seq(const std::vector<double>& a) {
    double s = 0.0;
    for (double x : a) s += x;
    return s;
}

double sum_reduction(const std::vector<double>& a, int threads) {
    const long n = static_cast<long>(a.size());
    const double* p = a.data();
    double s = 0.0;
#pragma omp parallel for reduction(+ : s) num_threads(threads)
    for (long i = 0; i < n; ++i) s += p[i];
    return s;
}

// atomic: аппаратная атомарная операция над одной ячейкой памяти
// (для double — цикл compare-and-swap). Все потоки бьются за одну кэш-линию.
double sum_atomic(const std::vector<double>& a, int threads) {
    const long n = static_cast<long>(a.size());
    const double* p = a.data();
    double s = 0.0;
#pragma omp parallel for num_threads(threads)
    for (long i = 0; i < n; ++i) {
#pragma omp atomic
        s += p[i];
    }
    return s;
}

double sum_atomic_local(const std::vector<double>& a, int threads) {
    const long n = static_cast<long>(a.size());
    const double* p = a.data();
    double s = 0.0;
#pragma omp parallel num_threads(threads)
    {
        double local = 0.0;
#pragma omp for nowait
        for (long i = 0; i < n; ++i) local += p[i];
#pragma omp atomic
        s += local;
    }
    return s;
}

// critical: блок кода, который в каждый момент выполняет не более одного потока.
// Реализован через мьютекс — дороже atomic, зато внутри может быть любой код.
double sum_critical(const std::vector<double>& a, int threads) {
    const long n = static_cast<long>(a.size());
    const double* p = a.data();
    double s = 0.0;
#pragma omp parallel for num_threads(threads)
    for (long i = 0; i < n; ++i) {
#pragma omp critical
        s += p[i];
    }
    return s;
}

double sum_critical_local(const std::vector<double>& a, int threads) {
    const long n = static_cast<long>(a.size());
    const double* p = a.data();
    double s = 0.0;
#pragma omp parallel num_threads(threads)
    {
        double local = 0.0;
#pragma omp for nowait
        for (long i = 0; i < n; ++i) local += p[i];
#pragma omp critical
        s += local;
    }
    return s;
}

// omp_lock_t: явный замок. По смыслу как critical, но им управляют функциями,
// и можно завести несколько независимых замков (например, по одному на ячейку данных).
double sum_lock(const std::vector<double>& a, int threads) {
    const long n = static_cast<long>(a.size());
    const double* p = a.data();
    double s = 0.0;
    omp_lock_t lock;
    omp_init_lock(&lock);
#pragma omp parallel for num_threads(threads)
    for (long i = 0; i < n; ++i) {
        omp_set_lock(&lock);
        s += p[i];
        omp_unset_lock(&lock);
    }
    omp_destroy_lock(&lock);
    return s;
}

double sum_lock_local(const std::vector<double>& a, int threads) {
    const long n = static_cast<long>(a.size());
    const double* p = a.data();
    double s = 0.0;
    omp_lock_t lock;
    omp_init_lock(&lock);
#pragma omp parallel num_threads(threads)
    {
        double local = 0.0;
#pragma omp for nowait
        for (long i = 0; i < n; ++i) local += p[i];
        omp_set_lock(&lock);
        s += local;
        omp_unset_lock(&lock);
    }
    omp_destroy_lock(&lock);
    return s;
}

int main(int argc, char** argv) {
    // синхронизация на каждой итерации очень медленная — размеры меньше, чем в задачах 1–3
    auto args = bench::parse_args(argc, argv, {10'000, 100'000, 1'000'000, 10'000'000});
    bench::csv_header();

    struct Variant { const char* name; double (*fn)(const std::vector<double>&, int); };
    const Variant variants[] = {
        {"reduction", sum_reduction},
        {"atomic", sum_atomic},
        {"atomic_local", sum_atomic_local},
        {"critical", sum_critical},
        {"critical_local", sum_critical_local},
        {"lock", sum_lock},
        {"lock_local", sum_lock_local},
    };

    for (long n : args.sizes) {
        auto a = bench::random_vector(n);
        double ref = sum_seq(a);
        volatile double seq_sink;
        bench::run_seq("07", "sequential", n, args, [&] { seq_sink = sum_seq(a); });
        (void)seq_sink;
        for (const auto& v : variants) {
            bench::check(v.name, v.fn(a, 4), ref, 1e-6);
            volatile double sink;
            bench::run_series("07", v.name, n, args, [&](int t) { sink = v.fn(a, t); });
            (void)sink;
        }
    }
}
