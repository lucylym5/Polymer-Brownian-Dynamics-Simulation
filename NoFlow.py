""" 
    采用Hookean弹簧的Rouse模型, 无流动, 不考虑排斥体积效应

    遍历N∈[20,90],  绘制log(sqrt{<Rg^2>})-logN关系
    对结果进行线性拟合，把拟合后的直线画在原图上，并在旁边标记斜率

    绘制log(tau)-logN关系,
    同样对结果进行线性拟合，把拟合后的直线画在原图上，并在旁边标记斜率

    默认 M=500、dt=0.05; 每个 N 推进约 3*tau_theory,
    在 0.2*tau_theory 到 1.5*tau_theory 区间拟合松弛时间
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib

import Necklace_chain as nc
import observables as ob
import parameters as p
import time_stepping as ts


def fit_power_law(N, values):
    """对 log(values)-log(N) 作直线拟合。"""
    x = np.log(np.asarray(N, dtype=float))
    y = np.log(np.asarray(values, dtype=float))
    coefficient, covariance = np.polyfit(x, y, 1, cov=True)
    residual = y - np.polyval(coefficient, x)
    return {
        "slope": float(coefficient[0]),
        "intercept": float(coefficient[1]),
        "slope_se": float(np.sqrt(covariance[0, 0])),
        "r_squared": float(1 - np.sum(residual**2) / np.sum((y - y.mean())**2)),
    }


def simulate_chain_length(N, config):
    M, dt = config["chains"], config["dt"]
    kBT, H, gamma = config["kBT"], config["H"], config["gamma"]
    rng = np.random.default_rng(
        np.random.SeedSequence(config["seed"], spawn_key=(int(N),))
    )
    r = nc.initialize_chain("Hookean", M, N, kBT, H, sigma=1.0, rng=rng)
    velocity_gradient = np.zeros((3, 3))
    tau_theory = float(ob.theoretical_relaxation_time(N, H, gamma))
    n_steps = int(np.ceil(config["duration"] * tau_theory / dt))
    save_every = max(1, int(round(tau_theory / (config["samples_per_tau"] * dt))))
    time = []
    modes = []
    rg2_sum_per_chain = np.zeros(M)

    def record(step):
        nonlocal rg2_sum_per_chain
        time.append(step * dt)
        modes.append(ob.first_rouse_mode(r))
        rg2, _, _ = ob.obervables(r)
        rg2_sum_per_chain += rg2

    record(0)
    for step in range(1, n_steps + 1):
        r = ts.euler_maruyama_step(
            "Hookean", r, velocity_gradient, kBT, gamma, H,
            1.0, dt, 1.0, 1.0, rng,  # Hookean 力不使用 Q0_square。
        )
        if step % save_every == 0 or step == n_steps:
            record(step)

    time = np.asarray(time)
    rg2_per_chain = rg2_sum_per_chain / len(time)
    rg2_mean = float(rg2_per_chain.mean())
    rg_rms = float(np.sqrt(rg2_mean))
    rg2_sem = float(rg2_per_chain.std(ddof=1) / np.sqrt(M))
    correlation = ob.mode_correlation(np.asarray(modes))
    fit_start_time = config["fit_start"] * tau_theory
    fit_end_time = config["fit_end"] * tau_theory
    selected = (time >= fit_start_time) & (time <= fit_end_time)
    tau = np.nan
    fit_status = "no_resolved_decay"
    if np.all(correlation[selected] > 0):
        candidate = ob.fit_relaxation_time(
            time, correlation, fit_start_time, fit_end_time,
        )
        if np.isfinite(candidate) and candidate > 0:
            tau = float(candidate)
            fit_status = "positive_decay_fit"
    # 整段窗口无有效衰减时，不用理论时间代替测量值。
    rg2_theory = float(3 * kBT / H * (N**2 - 1) / (6 * N))
    return {
        "N": int(N),
        "rg2_mean": rg2_mean,
        "rg2_sem": rg2_sem,
        "rg_rms": rg_rms,
        "rg_rms_sem": rg2_sem / (2 * rg_rms),
        "rg_rms_theory": float(np.sqrt(rg2_theory)),
        "tau": tau,
        "tau_theory": tau_theory,
        "total_time": float(time[-1]),
        "fit_start_time": fit_start_time,
        "fit_end_time": fit_end_time,
        "time_over_tau_theory": float(time[-1] / tau_theory),
        "n_steps": n_steps,
        "save_every": save_every,
        "n_records": len(time),
        "correlation_at_end": float(correlation[-1]),
        "tau_fit_status": fit_status,
        "time": time,
        "correlation": correlation,
    }


def plot_results(results, fits, config, output_dir, show=False):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = np.array([row["N"] for row in results])
    x = np.log(N)
    x_line = np.linspace(x.min(), x.max(), 200)
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.6), constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )
    panels = [
        ("rg", "rg_rms", "rg_rms_theory",
         r"$\log R_g$", "Equilibrium single-chain size"),
        ("tau", "tau", "tau_theory",
         r"$\log\tau_R$", "Longest mode correlation time"),
    ]
    for ax, (key, measured, theory, ylabel, title) in zip(axes, panels):
        values = np.array([row[measured] for row in results])
        valid = np.isfinite(values) & (values > 0)
        ax.plot(x[valid], np.log(values[valid]), "o", label="Simulation")
        ax.plot(
            x, np.log([row[theory] for row in results]), "--",
            color="0.4", label="Finite-N theory", zorder=3,
        )
        fit = fits[key]
        if fit is not None:
            ax.plot(
                x_line, fit["slope"] * x_line + fit["intercept"],
                color="tab:orange", label="Linear fit",
            )
            note = (
                f'Slope = {fit["slope"]:.3f}\n'
                f'R$^2$ = {fit["r_squared"]:.4f}'
            )
        else:
            note = "Insufficient decaying correlations for a fit"
        ax.text(
            0.04, 0.96, note, transform=ax.transAxes, va="top", fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )
        ax.set(xlabel=r"$\log N$", ylabel=ylabel, title=title)
        # 固定的是绘图区的物理长宽比；两幅图的数据单位无需等比例。
        ax.set_box_aspect(3 / 4)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f'M={config["chains"]}, dt={config["dt"]:g}, '
        f'T={config["duration"]:g} tau_theory per N',
        fontsize=12,
    )
    fig.canvas.draw()
    geometry = [
        {"width": float(ax.get_window_extent().width),
         "height": float(ax.get_window_extent().height)}
        for ax in axes
    ]
    fig.savefig(output_dir / "scaling_laws.png", dpi=200)

    fig_c, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    for row in results:
        ax.plot(
            row["time"] / row["tau_theory"], row["correlation"],
            label=f'N={row["N"]}',
        )
    scaled_time = np.linspace(0, config["duration"], 300)
    ax.plot(scaled_time, np.exp(-scaled_time), "k--", label=r"$e^{-t/\tau_{\rm theory}}$", zorder=3)
    ax.axvspan(config["fit_start"], config["fit_end"], color="grey", alpha=0.1)
    ax.set(
        xlabel=r"$t/\tau_{\rm theory}$", ylabel=r"$C_1(t)$",
        title="First-mode correlations and fitting window",
        xlim=(0, config["duration"]),
    )
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.25)
    fig_c.savefig(output_dir / "mode_correlations.png", dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    plt.close(fig_c)
    return geometry


def save_results(results, fits, config, output_dir, geometry):
    fields = [key for key in results[0] if key not in ("time", "correlation")]
    with (output_dir / "scaling_results.csv").open(
        "w", newline="", encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in results)
    curves = {}
    for row in results:
        curves[f'time_N{row["N"]}'] = row["time"]
        curves[f'correlation_N{row["N"]}'] = row["correlation"]
    np.savez_compressed(output_dir / "mode_correlations.npz", **curves)
    summary = {
        "config": config,
        "fits": fits,
        "axes_sizes_pixels_at_figure_dpi": geometry,
        "notes": [
            "Left panel fits log(sqrt(mean(Rg^2))) against log(N).",
            "Right panel fits log(measured tau) against log(N).",
            "Each N runs for about duration*tau_theory; fit-start and fit-end are in units of tau_theory.",
            "The theory only sets run length, sampling, fit window, and reference curves; bead dynamics determine the measured values.",
            "Both plot boxes have width:height=4:3 and the same dimensions.",
            "A non-decaying or non-positive correlation fit is omitted, never replaced by theory.",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-min", type=int, default=20)
    parser.add_argument("--n-max", type=int, default=90)
    parser.add_argument("--n-step", type=int, default=10)
    parser.add_argument("--chains", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--duration", type=float, default=3.0, help="总时间 / 理论相关时间")
    parser.add_argument("--samples-per-tau", type=int, default=100)
    parser.add_argument("--fit-start", type=float, default=0.2, help="拟合起点 / 理论相关时间")
    parser.add_argument("--fit-end", type=float, default=1.5, help="拟合终点 / 理论相关时间")
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=p.GAMMA)
    parser.add_argument("--kBT", type=float, default=p.KB_DIMLESS * p.TEMP_DIMLESS)
    parser.add_argument("--seed", type=int, default=p.RANDOM_SEED)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "noFlow_results",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not (2 <= args.n_min <= args.n_max and args.n_step > 0):
        parser.error("需要 2 <= n-min <= n-max，且 n-step > 0。")
    bead_numbers = list(range(args.n_min, args.n_max + 1, args.n_step))
    if bead_numbers[-1] != args.n_max:
        bead_numbers.append(args.n_max)
    if len(bead_numbers) < 3:
        parser.error("拟合至少需要三个不同的珠子数。")
    if min(args.H, args.gamma, args.kBT, args.dt, args.duration) <= 0:
        parser.error("物理参数、dt 和 duration 必须为正。")
    if args.dt >= args.gamma / (2 * args.H):
        parser.error("显式胡克推进需要 dt < gamma/(2H)。")
    if not (0 <= args.fit_start < args.fit_end <= args.duration):
        parser.error("需要 0 <= fit-start < fit-end <= duration。")
    if (args.fit_end - args.fit_start) * args.samples_per_tau < 2:
        parser.error("拟合窗口至少需要三个记录时刻。")
    if args.chains < 2 or args.workers < 1 or args.samples_per_tau < 10:
        parser.error("需要 chains >= 2、workers >= 1、samples-per-tau >= 10。")
    config = {
        key: value for key, value in vars(args).items()
        if key not in ("output", "show")
    }
    config["bead_numbers"] = bead_numbers
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f'N={bead_numbers}; M={args.chains}; dt={args.dt}; '
        f'T={args.duration:g}*tau_theory per N; '
        f'fit={args.fit_start:g}..{args.fit_end:g} tau_theory',
        flush=True,
    )
    results = []

    def collect(row):
        results.append(row)
        print(
            f'N={row["N"]:3d}: Rg={row["rg_rms"]:.5g}, '
            f'tau_fit={row["tau"]:.5g}, '
            f'T={row["total_time"]:.5g}',
            flush=True,
        )

    if args.workers == 1:
        for N in bead_numbers:
            collect(simulate_chain_length(N, config))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(simulate_chain_length, N, config) for N in bead_numbers]
            for future in as_completed(futures):
                collect(future.result())
    results.sort(key=lambda row: row["N"])
    measured_tau = np.array([row["tau"] for row in results])
    valid = np.isfinite(measured_tau) & (measured_tau > 0)
    fits = {
        "rg": fit_power_law(bead_numbers, [row["rg_rms"] for row in results]),
        "tau": (
            fit_power_law(np.array(bead_numbers)[valid], measured_tau[valid])
            if valid.sum() >= 3 else None
        ),
    }
    geometry = plot_results(results, fits, config, args.output, args.show)
    save_results(results, fits, config, args.output, geometry)
    print(f'Rg slope: {fits["rg"]["slope"]:.5f}', flush=True)
    if fits["tau"] is not None:
        print(f'tau slope: {fits["tau"]["slope"]:.5f}', flush=True)
    print(f"Results: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
