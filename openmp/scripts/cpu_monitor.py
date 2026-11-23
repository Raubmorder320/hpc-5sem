#!/usr/bin/env python3
"""Запускает программу и каждые --interval секунд записывает загрузку каждого логического CPU
(/proc/stat), а также число потоков и суммарную загрузку CPU процесса (/proc/<pid>).

Пример:
    python3 scripts/cpu_monitor.py --out results/monitor/task06 -- ./bin/task06 --sizes 100000 --threads 16

Результат:
    <out>_cpu.csv   — time_s, cpu0 ... cpuN  (загрузка каждого CPU, %)
    <out>_proc.csv  — time_s, threads, proc_cpu_pct  (100% = одно ядро целиком)
    <out>.png       — тепловая карта загрузки CPU во времени + график процесса
Вывод самой программы (CSV с замерами) идёт в stdout как обычно.
"""
import argparse
import os
import subprocess
import sys
import time

CLK_TCK = os.sysconf("SC_CLK_TCK")


def read_cpu_times():
    """{cpu_index: (busy_ticks, total_ticks)} по строкам cpuN из /proc/stat."""
    res = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            parts = line.split()
            vals = list(map(int, parts[1:9]))  # user nice system idle iowait irq softirq steal
            idle = vals[3] + vals[4]
            total = sum(vals)
            res[int(parts[0][3:])] = (total - idle, total)
    return res


def read_proc(pid):
    """(ticks utime+stime, число потоков) процесса или None, если он уже завершился."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            # имя процесса в скобках может содержать пробелы — режем по последней ')'
            fields = f.read().rsplit(")", 1)[1].split()
        ticks = int(fields[11]) + int(fields[12])  # utime, stime
        with open(f"/proc/{pid}/status") as f:
            threads = next(int(l.split()[1]) for l in f if l.startswith("Threads:"))
        return ticks, threads
    except (FileNotFoundError, ProcessLookupError, StopIteration):
        return None


def plot(out, cpu_rows, proc_rows, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.array([r[0] for r in cpu_rows])
    load = np.array([r[1:] for r in cpu_rows]).T  # [cpu][time]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.3]})
    im = ax1.imshow(load, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=100,
                    extent=[t[0], t[-1], -0.5, load.shape[0] - 0.5], interpolation="nearest")
    ax1.set_ylabel("логический CPU")
    ax1.set_yticks(range(load.shape[0]))
    ax1.set_title(title, fontsize=10)
    # шкала цвета как вложенная ось справа — не отнимает ширину, графики остаются выровнены
    fig.colorbar(im, cax=ax1.inset_axes([1.01, 0, 0.015, 1]), label="загрузка CPU, %")

    pt = [r[0] for r in proc_rows]
    ax2.plot(pt, [r[2] for r in proc_rows], color="tab:blue")
    ax2.set_ylabel("CPU процесса, %", color="tab:blue")
    ax2.set_xlabel("время, с")
    ax2b = ax2.twinx()
    ax2b.plot(pt, [r[1] for r in proc_rows], color="tab:orange")
    ax2b.set_ylabel("потоков", color="tab:orange")
    ax2.grid(alpha=0.3)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.95, bottom=0.08, hspace=0.12)
    fig.savefig(out + ".png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--out", required=True, help="префикс выходных файлов")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("не указана команда после --")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cpu_rows, proc_rows = [], []
    prev_cpu = read_cpu_times()
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd)
    prev_proc = read_proc(proc.pid)
    prev_t = t0
    while proc.poll() is None:
        time.sleep(args.interval)
        now = time.monotonic()
        cur_cpu = read_cpu_times()
        row = [round(now - t0, 3)]
        for c in sorted(cur_cpu):
            db = cur_cpu[c][0] - prev_cpu[c][0]
            dt = cur_cpu[c][1] - prev_cpu[c][1]
            row.append(round(100.0 * db / dt, 1) if dt else 0.0)
        cpu_rows.append(row)
        prev_cpu = cur_cpu

        cur_proc = read_proc(proc.pid)
        if cur_proc and prev_proc:
            pct = 100.0 * (cur_proc[0] - prev_proc[0]) / CLK_TCK / (now - prev_t)
            proc_rows.append((round(now - t0, 3), cur_proc[1], round(pct, 1)))
        prev_proc, prev_t = cur_proc, now

    ncpu = len(cpu_rows[0]) - 1 if cpu_rows else 0
    with open(args.out + "_cpu.csv", "w") as f:
        f.write("time_s," + ",".join(f"cpu{i}" for i in range(ncpu)) + "\n")
        for r in cpu_rows:
            f.write(",".join(map(str, r)) + "\n")
    with open(args.out + "_proc.csv", "w") as f:
        f.write("time_s,threads,proc_cpu_pct\n")
        for r in proc_rows:
            f.write(",".join(map(str, r)) + "\n")

    if len(cpu_rows) >= 2:
        plot(args.out, cpu_rows, proc_rows, " ".join(cmd))
        peak = max(r[2] for r in proc_rows) if proc_rows else 0
        print(f"[monitor] {len(cpu_rows)} замеров, пик CPU процесса {peak:.0f}%, "
              f"макс. потоков {max((r[1] for r in proc_rows), default=0)} -> {args.out}.png",
              file=sys.stderr)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
