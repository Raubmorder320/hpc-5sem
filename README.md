# Введение в суперкомпьютерные вычисления — OpenMP

Решения заданий OpenMP 1–9 на C++.

- [Условия заданий](HPC-tasks-2026.pdf)
- [Отчёт с графиками и выводами](openmp/REPORT.md)
- [Графики и полные таблицы](openmp/results/plots/)
- [Данные замеров](openmp/results/)

| № | Задача | Исходник | Замеры |
| --- | --- | --- | --- |
| 01 | Минимум вектора с редукцией и без неё | [Код](openmp/task01_minmax/main.cpp) | [CSV](openmp/results/task01.csv) · [таблица](openmp/results/plots/task01.md) |
| 02 | Скалярное произведение | [Код](openmp/task02_dot/main.cpp) | [CSV](openmp/results/task02.csv) · [таблица](openmp/results/plots/task02.md) |
| 03 | Интегрирование методом прямоугольников | [Код](openmp/task03_integral/main.cpp) | [CSV](openmp/results/task03.csv) · [таблица](openmp/results/plots/task03.md) |
| 04 | Максимум из минимумов строк | [Код](openmp/task04_maxmin/main.cpp) | [CSV](openmp/results/task04.csv) · [таблица](openmp/results/plots/task04.md) |
| 05 | Треугольная и ленточная матрицы | [Код](openmp/task05_special_matrix/main.cpp) | [CSV](openmp/results/task05.csv) · [таблица](openmp/results/plots/task05.md) |
| 06 | Распределение неравномерной нагрузки | [Код](openmp/task06_schedule/main.cpp) | [CSV](openmp/results/task06.csv) · [таблица](openmp/results/plots/task06.md) |
| 07 | Сравнение способов синхронизации | [Код](openmp/task07_sync/main.cpp) | [CSV](openmp/results/task07.csv) · [таблица](openmp/results/plots/task07.md) |
| 08 | Чтение и вычисление через sections | [Код](openmp/task08_sections/main.cpp) | [CSV](openmp/results/task08.csv) · [таблица](openmp/results/plots/task08.md) |
| 09 | Вложенный параллелизм | [Код](openmp/task09_nested/main.cpp) | [CSV](openmp/results/task09.csv) · [таблица](openmp/results/plots/task09.md) |

## Сборка и быстрый запуск

Нужны Linux, GNU Make и GCC с поддержкой C++20 и OpenMP. Для сборки Python не требуется.
Из корня репозитория выполните:

```bash
cd openmp
make -j4
./bin/task01 --sizes 1000,100000 --threads 1,2,4 --reps 3
./bin/task09 --sizes 100,1000 --threads 1,2,4 --reps 3
```

Для задачи 8 сначала подготовьте входной файл. Последние два числа у `--gen` задают число пар и длину вектора.

```bash
mkdir -p data
./bin/task08 --gen data/example.bin 20 10000
./bin/task08 --file data/example.bin --threads 1,2,4 --reps 3
```

Программы печатают CSV в стандартный вывод. В задаче 8 `--threads` задаёт размер внутренней команды вычисления, чтение выполняет отдельный поток.

## Воспроизведение экспериментов

Из каталога `openmp`. Для графиков нужны Python 3, matplotlib и pandas.

```bash
./scripts/run_all.sh          # все девять задач
./scripts/run_all.sh 05 06    # только выбранные задачи
```

Новые CSV, таблицы и графики сохраняются в `openmp/runs/`. Данные отчёта хранятся в `openmp/results/`.
Полный прогон занимает несколько минут и создаёт около 1.9 ГБ входных файлов для задачи 8.
Крупнейший тест задачи 2 выделяет около 1.6 ГБ оперативной памяти. Для быстрых проверок используйте примеры выше.
В задаче 9 вложенные варианты измеряются при размере матрицы до 1000 и до 16 потоков.
