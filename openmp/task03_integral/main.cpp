// OpenMP-3. Определённый интеграл методом прямоугольников:
//   y ≈ h * sum_{i=0}^{N-1} f(x_i),  x_i = a + i*h,  h = (b - a) / N
// n из --sizes — число разбиений N.
#include "bench.hpp"

#include <cmath>

double f(double x) {
    return std::sin(x);  // ∫_0^π sin(x) dx = 2 — удобно проверять точность
}

double integrate_seq(double a, double b, long n) {
    const double h = (b - a) / n;
    double s = 0.0;
    for (long i = 0; i < n; ++i) s += f(a + i * h);
    return s * h;
}

// x_i вычисляется из i, а не накапливается (x += h), поэтому итерации независимы
// и их можно раздать потокам в любом порядке.
double integrate_par(double a, double b, long n, int threads) {
    const double h = (b - a) / n;
    double s = 0.0;
#pragma omp parallel for reduction(+ : s) num_threads(threads)
    for (long i = 0; i < n; ++i) s += f(a + i * h);
    return s * h;
}

int main(int argc, char** argv) {
    auto args = bench::parse_args(argc, argv, {1'000, 100'000, 10'000'000, 100'000'000});
    bench::csv_header();

    const double a = 0.0, b = M_PI;
    for (long n : args.sizes) {
        double y = integrate_par(a, b, n, 4);
        bench::check("integral", y, integrate_seq(a, b, n), 1e-6);
        std::fprintf(stderr, "N=%ld: y=%.12f, |y-2|=%.3e\n", n, y, std::abs(y - 2.0));

        volatile double sink;
        bench::run_series("03", "reduction", n, args, [&](int t) { sink = integrate_par(a, b, n, t); });
        (void)sink;
    }
}
