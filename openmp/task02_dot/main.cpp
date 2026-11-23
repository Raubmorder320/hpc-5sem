// OpenMP-2. Скалярное произведение двух векторов.
#include "bench.hpp"

double dot_seq(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// Итерации делятся между потоками блоками (schedule static по умолчанию),
// у каждого потока своя частичная сумма s, в конце они складываются.
double dot_par(const std::vector<double>& a, const std::vector<double>& b, int threads) {
    const long n = static_cast<long>(a.size());
    const double* pa = a.data();
    const double* pb = b.data();
    double s = 0.0;
#pragma omp parallel for reduction(+ : s) num_threads(threads)
    for (long i = 0; i < n; ++i) s += pa[i] * pb[i];
    return s;
}

int main(int argc, char** argv) {
    auto args = bench::parse_args(argc, argv, {1'000, 100'000, 10'000'000, 100'000'000});
    bench::csv_header();

    for (long n : args.sizes) {
        auto a = bench::random_vector(n, 1);
        auto b = bench::random_vector(n, 2);
        // порядок сложения меняется, поэтому допуск больше
        bench::check("dot", dot_par(a, b, 4), dot_seq(a, b), 1e-6);

        volatile double sink;
        bench::run_series("02", "reduction", n, args, [&](int t) { sink = dot_par(a, b, t); });
        (void)sink;
    }
}
