#!/usr/bin/env bash
# Полный прогон всех задач с параметрами по умолчанию: CSV в runs/, графики в runs/plots/.
# Запуск из папки openmp:  ./scripts/run_all.sh
# Отдельные задачи:        ./scripts/run_all.sh 04 06
set -euo pipefail
cd "$(dirname "$0")/.."

make -s
# Пониженный приоритет для длительных замеров.
run=(nice -n 10)
export RESULTS_DIR="${RESULTS_DIR:-runs}"
mkdir -p "$RESULTS_DIR" data

tasks=("$@")
[ ${#tasks[@]} -eq 0 ] && tasks=(01 02 03 04 05 06 07 08 09)

for t in "${tasks[@]}"; do
    echo "=== task$t ===" >&2
    if [ "$t" = "08" ]; then
        # одинаковый объём (640 МБ) при разной длине векторов: 10^4, 10^5, 10^6
        [ -f data/v1e4.bin ] || ./bin/task08 --gen data/v1e4.bin 4000 10000
        [ -f data/vectors.bin ] || ./bin/task08 --gen data/vectors.bin 400 100000
        [ -f data/v1e6.bin ] || ./bin/task08 --gen data/v1e6.bin 40 1000000
        {
            "${run[@]}" ./bin/task08 --file data/v1e4.bin --reps 9
            "${run[@]}" ./bin/task08 --file data/vectors.bin --reps 9 | tail -n +2
            "${run[@]}" ./bin/task08 --file data/v1e6.bin --reps 9 | tail -n +2
        } > "$RESULTS_DIR/task08.csv"
    else
        reps=5; [ "$t" = "02" ] && reps=15  # задача 2 шумная — больше повторов
        "${run[@]}" ./bin/task$t --reps $reps > "$RESULTS_DIR/task$t.csv"
    fi
done

python3 scripts/plot.py
