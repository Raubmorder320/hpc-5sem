// OpenMP-8. Скалярные произведения для последовательности пар векторов из файла.
// Чтение и вычисление — две раздельные задачи в #pragma omp parallel sections
// (двойная буферизация: пока считается одна пара, читается следующая).
//
// Подготовка данных:  ./bin/task08 --gen data/vectors.bin <pairs> <dim>
// Эксперимент:        ./bin/task08 --file data/vectors.bin [--threads ...] [--reps ...]
#include "bench.hpp"

#include <cstdint>
#include <cstring>
#include <string>

static bool valid_shape(long pairs, long dim) {
    return pairs > 0 && pairs < std::numeric_limits<long>::max() &&
           dim > 0 && dim <= std::numeric_limits<long>::max() / 2;
}

// Формат файла (бинарный): int64 pairs, int64 dim, затем pairs * 2 * dim значений double.
struct Header {
    int64_t pairs;
    int64_t dim;
};

void generate_file(const std::string& path, long pairs, long dim) {
    if (!valid_shape(pairs, dim)) {
        std::fprintf(stderr, "неверное число пар или длина вектора\n");
        std::exit(1);
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror(path.c_str()); std::exit(1); }
    Header h{pairs, dim};
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1;
    for (long k = 0; ok && k < pairs; ++k) {
        auto a = bench::random_vector(dim, 2 * k + 1);
        auto b = bench::random_vector(dim, 2 * k + 2);
        ok = std::fwrite(a.data(), sizeof(double), dim, f) == static_cast<size_t>(dim) &&
             std::fwrite(b.data(), sizeof(double), dim, f) == static_cast<size_t>(dim);
    }
    if (std::fclose(f) != 0 || !ok) {
        std::fprintf(stderr, "не удалось записать %s\n", path.c_str());
        std::exit(1);
    }
    std::fprintf(stderr, "записано %ld пар по %ld элементов (%.1f МБ) в %s\n",
                 pairs, dim, 16.0 * pairs * dim / 1e6, path.c_str());
}

static FILE* open_vectors(const std::string& path, Header& h) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f || std::fread(&h, sizeof h, 1, f) != 1) { std::perror(path.c_str()); std::exit(1); }
    if (!valid_shape(h.pairs, h.dim)) {
        std::fprintf(stderr, "неверный заголовок файла %s\n", path.c_str());
        std::exit(1);
    }
    return f;
}

// Пара (a, b) лежит в одном буфере подряд: a = buf[0..dim), b = buf[dim..2*dim).
static void read_pair(FILE* f, double* buf, long dim) {
    if (std::fread(buf, sizeof(double), 2 * dim, f) != static_cast<size_t>(2 * dim)) {
        std::fprintf(stderr, "файл короче, чем указано в заголовке\n");
        std::exit(1);
    }
}

static double dot(const double* a, const double* b, long n, int threads) {
    double s = 0.0;
#pragma omp parallel for reduction(+ : s) num_threads(threads)
    for (long i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// Эталон: прочитать пару — посчитать — прочитать следующую. Чтение и счёт не перекрываются.
std::vector<double> dots_seq(const std::string& path) {
    Header h;
    FILE* f = open_vectors(path, h);
    std::vector<double> buf(2 * h.dim), res(h.pairs);
    for (long k = 0; k < h.pairs; ++k) {
        read_pair(f, buf.data(), h.dim);
        res[k] = dot(buf.data(), buf.data() + h.dim, h.dim, 1);
    }
    std::fclose(f);
    return res;
}

// Двойная буферизация: пока одна секция считает пару k-1 из буфера (k-1)%2,
// другая читает пару k в буфер k%2. Синхронизация — неявный барьер в конце sections
// (п. 5.6 лекции: «выполнение директивы sections синхронизировано»): к следующему шагу
// прочитанная пара уже в памяти, а буфер посчитанной пары свободен для чтения в него.
// Параллельная область одна на весь расчёт: её 2 потока проходят цикл по шагам,
// а директива sections на каждом шаге раздаёт им две секции.
// threads — потоки для вложенного parallel for внутри секции вычислений.
std::vector<double> dots_sections(const std::string& path, int threads) {
    Header h;
    FILE* f = open_vectors(path, h);
    const long pairs = h.pairs, dim = h.dim;

    std::vector<double> buf[2] = {std::vector<double>(2 * dim), std::vector<double>(2 * dim)};
    std::vector<double> res(pairs);

    omp_set_nested(1);  // разрешить вложенный parallel for внутри секции (п. 5.7.7)
#pragma omp parallel num_threads(2)
    for (long k = 0; k <= pairs; ++k) {  // k объявлена в регионе => своя у каждого потока
#pragma omp sections
        {
#pragma omp section
            {  // ввод: пара k
                if (k < pairs) read_pair(f, buf[k % 2].data(), dim);
            }
#pragma omp section
            {  // вычисление: пара k-1
                if (k > 0) {
                    const double* b = buf[(k - 1) % 2].data();
                    res[k - 1] = dot(b, b + dim, dim, threads);
                }
            }
        }  // неявный барьер: оба потока ждут друг друга перед следующим шагом
    }
    std::fclose(f);
    return res;
}

int main(int argc, char** argv) {
    if (argc >= 5 && std::strcmp(argv[1], "--gen") == 0) {
        generate_file(argv[2], std::atol(argv[3]), std::atol(argv[4]));
        return 0;
    }
    if (argc < 3 || std::strcmp(argv[1], "--file") != 0) {
        std::fprintf(stderr, "usage: %s --gen <path> <pairs> <dim>\n"
                             "       %s --file <path> [--threads ...] [--reps ...]\n", argv[0], argv[0]);
        return 1;
    }
    std::string path = argv[2];
    auto args = bench::parse_args(argc - 2, argv + 2, {1});
    bench::csv_header();

    auto ref = dots_seq(path);
    auto got = dots_sections(path, 4);
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "[FAIL] sections: %zu results, expected %zu\n", got.size(), ref.size());
        return 1;
    }
    for (size_t i = 0; i < ref.size(); ++i) bench::check("sections", got[i], ref[i], 1e-6);

    // n в CSV — длина вектора (объём файла во всех экспериментах одинаковый, меняется только она)
    Header h;
    std::fclose(open_vectors(path, h));
    bench::run_seq("08", "sequential", h.dim, args, [&] { dots_seq(path); });
    bench::run_series("08", "sections", h.dim, args, [&](int t) { dots_sections(path, t); });
}
