// OpenMP-6. Исследование режимов schedule (static, dynamic, guided)
// на цикле с неравномерной нагрузкой итераций.
// n из --sizes — число итераций цикла.
#include "bench.hpp"

#include <random>
#include <string>

constexpr int HEAVY_DRAWS = 1000;  // сколько случайных чисел генерирует «тяжёлая» итерация

// Итерация i «тяжёлая» с вероятностью i/n: к концу цикла тяжёлых итераций всё больше.
// Генератор засевается номером итерации, поэтому нагрузка одинакова при любом числе потоков
// и любом порядке выполнения — сравниваются только стратегии распределения.
double heavy_work(long i, long n) {
    std::minstd_rand rng(static_cast<unsigned>(i) + 1);
    if (static_cast<long>(rng() % n) >= i) return 1.0;  // лёгкая итерация
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double s = 0.0;
    for (int k = 0; k < HEAVY_DRAWS; ++k) s += dist(rng);
    return s;
}

double run_loop(long n, omp_sched_t sched, int chunk, int threads) {
    omp_set_schedule(sched, chunk);
    double s = 0.0;
#pragma omp parallel for schedule(runtime) reduction(+ : s) num_threads(threads)
    for (long i = 0; i < n; ++i) s += heavy_work(i, n);
    return s;
}

int main(int argc, char** argv) {
    auto args = bench::parse_args(argc, argv, {10'000, 100'000});
    bench::csv_header();

    struct Sched { const char* name; omp_sched_t kind; int chunk; };
    const Sched schedules[] = {
        {"static", omp_sched_static, 0},
        {"static_1", omp_sched_static, 1},
        {"static_64", omp_sched_static, 64},
        {"dynamic_1", omp_sched_dynamic, 1},
        {"dynamic_64", omp_sched_dynamic, 64},
        {"guided_1", omp_sched_guided, 1},
        {"guided_64", omp_sched_guided, 64},
    };

    for (long n : args.sizes) {
        double ref = run_loop(n, omp_sched_static, 0, 1);
        for (const auto& s : schedules) {
            bench::check(s.name, run_loop(n, s.kind, s.chunk, 4), ref, 1e-9);
            volatile double sink;
            bench::run_series("06", s.name, n, args, [&](int t) { sink = run_loop(n, s.kind, s.chunk, t); });
            (void)sink;
        }
    }
}
