#!/usr/bin/env python3
"""Строит графики времени и ускорения по CSV из results/ и таблицы для отчёта.

    python3 scripts/plot.py                 # все results/task*.csv
    python3 scripts/plot.py results/task06.csv

Для каждого CSV:
    results/plots/<task>_n<n>.png    — для размера n: слева время (лог. шкала), справа ускорение
    results/plots/<task>_by_size.png — ускорение от размера задачи при p = 4, 16, 32
    results/plots/<task>.md          — таблицы «потоки × вариант» (время, с / ускорение)

Ускорение. Если в CSV есть последовательная версия (вариант «sequential» или «<префикс>_sequential»),
ускорение всех вариантов считается относительно неё: S = T_посл / T(p). Тогда варианты сравнимы
между собой. Иначе — относительно того же варианта на 1 потоке: S = T(1) / T(p).
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

NCPU = os.cpu_count()


def size_label(x, _pos=None):
    """1000 -> 10³, 5000 -> 5000: степени десятки показываем степенью, остальное числом."""
    k = len(str(int(x))) - 1
    return f"$10^{{{k}}}$" if x == 10**k and k >= 3 else str(int(x))


def with_speedup(df):
    """Добавляет столбец s (ускорение) и возвращает (df без строк sequential, есть ли база, база по (вариант, n))."""
    seq_names = [v for v in df.variant.unique() if v.endswith("sequential")]
    if not seq_names:
        return df.assign(s=df.speedup), False, {}
    seq = df[df.variant.isin(seq_names) & (df.threads == 1)]
    base = {(r.variant, r.n): r.time_s for r in seq.itertuples()}

    def base_of(variant, n):
        # самая длинная подходящая база: triangular_static -> triangular_sequential
        name = max((s for s in seq_names if variant.startswith(s[: -len("sequential")])), key=len)
        return base[(name, n)]

    rest = df[~df.variant.isin(seq_names)].copy()
    rest["s"] = [base_of(r.variant, r.n) / r.time_s for r in rest.itertuples()]
    return rest, True, base


def plot_file(path, outdir):
    try:
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"{path}: пустой файл, пропущен")
        return
    if raw.empty:
        print(f"{path}: пустой файл, пропущен")
        return
    name = os.path.splitext(os.path.basename(path))[0]
    df, has_seq, base = with_speedup(raw)
    s_label = "ускорение T_посл / T(p)" if has_seq else "ускорение T(1) / T(p)"

    md = [f"# {name}\n"]
    if has_seq:
        md.append("Ускорение — относительно однопоточной базовой версии.\n")
    for n, g in df.groupby("n"):
        fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(12, 4.5))
        for variant, gv in g.groupby("variant", sort=False):
            gv = gv.sort_values("threads")
            ax_t.plot(gv.threads, gv.time_s, marker="o", label=variant)
            ax_s.plot(gv.threads, gv.s, marker="o", label=variant)
        for (seq_name, seq_n), t in base.items():
            if seq_n == n:
                ax_t.axhline(t, color="k", ls="--", lw=0.8)
        tmax = min(g.threads.max(), NCPU)
        ax_s.plot([1, tmax], [1, tmax], "k:", lw=0.8, label="идеал S = p")
        ax_s.axhline(1, color="k", lw=0.6)
        for ax in (ax_t, ax_s):
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(g.threads.unique()))
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.axvline(NCPU, color="gray", ls=":", lw=1)
            ax.set_xlabel("число потоков p")
            ax.grid(alpha=0.3)
        ax_t.set_yscale("log")
        # подписи и промежуточных делений, иначе на диапазоне ~1 декада остаётся одна подпись
        ax_t.yaxis.set_minor_formatter(mticker.LogFormatterSciNotation(minor_thresholds=(2, 0.5)))
        ax_t.set_ylabel("время, с" + ("  (штрих — последовательная)" if has_seq else ""))
        ax_s.set_ylabel(s_label)
        ax_s.legend(fontsize=8, ncol=2)
        fig.suptitle(f"{name}, n = {n}  (вертикальный пунктир — {NCPU} логических CPU)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_n{n}.png"), dpi=120)
        plt.close(fig)

        cell = g.assign(v=g.time_s.map("{:.5f}".format) + " / " + g.s.map("{:.2f}".format))
        table = cell.pivot(index="threads", columns="variant", values="v").fillna("—")
        table = table[list(dict.fromkeys(g.variant))]  # исходный порядок вариантов
        md.append(f"\n## n = {n}\n")
        for (seq_name, seq_n), t in base.items():
            if seq_n == n:
                md.append(f"{seq_name}: {t:.5f} с  ")
        md.append("\nвремя, с / ускорение\n")
        md.append("| потоки | " + " | ".join(table.columns) + " |")
        md.append("|---" * (len(table.columns) + 1) + "|")
        for th, row in table.iterrows():
            md.append(f"| {th} | " + " | ".join(row.values) + " |")
    with open(os.path.join(outdir, f"{name}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    if df.n.nunique() > 1:
        plot_by_size(df, name, outdir, s_label)
        if df.variant.nunique() == 1:
            plot_all_sizes(df, name, outdir, s_label)
    print(f"{path} -> {outdir}/{name}_*.png, {name}.md")


# Ускорение в зависимости от размера задачи при фиксированном числе потоков:
# p = 4 (меньше числа CPU), NCPU (равно), 2*NCPU (больше).
def plot_by_size(df, name, outdir, s_label):
    ps = [p for p in (4, NCPU, 2 * NCPU) if p in set(df.threads)]
    if not ps:
        ps = sorted(df.threads.unique())[:3]
    fig, axes = plt.subplots(1, len(ps), figsize=(5 * len(ps), 4.2), sharey=True, squeeze=False)
    for ax, p in zip(axes[0], ps):
        g = df[df.threads == p]
        for variant, gv in g.groupby("variant", sort=False):
            gv = gv.sort_values("n")
            ax.plot(gv.n, gv.s, marker="o", label=variant)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(sorted(df.n.unique()))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(size_label))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel("размер задачи n")
        ax.set_title(f"p = {p}")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(s_label)
    axes[0][-1].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{name}: ускорение от размера задачи")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}_by_size.png"), dpi=120)
    plt.close(fig)


# Один вариант решения (задачи 2, 3, 4, 8): ускорение от числа потоков, по линии на каждый размер.
def plot_all_sizes(df, name, outdir, s_label):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n, g in df.groupby("n"):
        g = g.sort_values("threads")
        ax.plot(g.threads, g.s, marker="o", label=f"n = {size_label(n)}")
    tmax = min(df.threads.max(), NCPU)
    if df.s.max() > 2:  # в задаче 8 ускорение < 2 — линия идеала только сжала бы график
        ax.plot([1, tmax], [1, tmax], "k:", lw=0.8, label="идеал S = p")
    ax.axhline(1, color="k", lw=0.6)
    ax.axvline(NCPU, color="gray", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df.threads.unique()))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("число потоков p")
    ax.set_ylabel(s_label)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.suptitle(f"{name}: ускорение для всех размеров задачи")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}_all_n.png"), dpi=120)
    plt.close(fig)


def main():
    results_dir = os.environ.get("RESULTS_DIR", "results")
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(results_dir, "task*.csv")))
    outdir = os.path.join(results_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    for path in files:
        plot_file(path, outdir)


if __name__ == "__main__":
    main()
