import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support, get_context
from pathlib import Path

import matplotlib

# 和参考程序一样直接保存图片；使用非交互后端，不弹出绘图窗口。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import Dumbell_ElongationalFlow as bd


SELECTED_WI = (0.1, 0.5, 1.0, 2.0)
DEFAULT_Q0_SQUARE = 100.0
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "Dumbell_ElongationalFlow_figures"


def hookean_qx2_analytical(time, Wi, kB, T, H, gamma, initial_qx2=None):
    """Hookean 连续随机方程的精确二阶矩，包含布朗噪声贡献。

    M' = rate*M + 4D, rate=(2Wi-1)/lambda, lambda=gamma/(4H)。
    默认 M(0)=kB*T/H，与原函数的平衡态初始化一致。
    使用 expm1 避免 Wi 接近 0.5 时相减损失精度。
    """
    time = np.asarray(time, dtype=float)
    physical_parameters = np.asarray([kB, T, H, gamma], dtype=float)
    if not np.all(np.isfinite(physical_parameters)) or np.any(physical_parameters <= 0):
        raise ValueError("kB, T, H and gamma must be positive and finite.")
    if not np.isfinite(Wi) or np.any(~np.isfinite(time)) or np.any(time < 0):
        raise ValueError("Wi must be finite; times must be finite and nonnegative.")

    initial = kB * T / H if initial_qx2 is None else float(initial_qx2)
    if not np.isfinite(initial) or initial < 0:
        raise ValueError("initial_qx2 must be finite and nonnegative.")
    D = kB * T / gamma
    relaxation_time = gamma / (4.0 * H)
    rate = (2.0 * Wi - 1.0) / relaxation_time
    if rate == 0.0:
        return initial + 4.0 * D * time
    with np.errstate(over="raise", invalid="raise"):
        return initial * np.exp(rate * time) + 4.0 * D * np.expm1(rate * time) / rate


def _simulate_one(model, Wi, q0_square, dt, total_time, particles):
    """所有构象演化直接调用原文件中的完整模拟函数。"""
    time, q2 = bd.dumbell_elongation(
        name=model,
        Q0_square=q0_square,
        kB=bd.BOLTZMANN_CONSTANT,
        T=bd.TEMPERATURE,
        gamma=bd.GAMMA,
        H=bd.SPRING_CONSTANT,
        Wi=float(Wi),
        dt=dt,
        total_time=total_time,
        particle_number=particles,
        dimension=3,
        random_seed=bd.RANDOM_SEED,
    )
    if not np.all(np.isfinite(q2)):
        raise FloatingPointError(f"Nonfinite second moments: {model}, Wi={Wi:g}.")
    if model == "FENE" and np.any(np.sum(q2, axis=1) >= q0_square):
        raise FloatingPointError(f"FENE mean squared length exceeded Q0^2 at Wi={Wi:g}.")
    return time, q2


def run_simulations(Wi_list, q0_square, dt, total_time, particles, workers=4):
    """计算全部 Wi 的完整均方时间序列，四个代表点直接复用扫描结果。"""
    jobs = [(model, i, float(Wi)) for model in ("Hookean", "FENE")
            for i, Wi in enumerate(Wi_list)]
    results = {"Hookean": [None] * len(Wi_list), "FENE": [None] * len(Wi_list)}
    time = None

    def record(model, index, Wi, result, completed):
        nonlocal time
        job_time, q2 = result
        if time is None:
            time = job_time
        elif not np.array_equal(time, job_time):
            raise ValueError("Simulation time grids are inconsistent.")
        results[model][index] = q2
        print(f"[{completed:02d}/{len(jobs)}] {model:7s} Wi={Wi:.2f}", flush=True)

    if workers == 1:
        for completed, (model, index, Wi) in enumerate(jobs, start=1):
            result = _simulate_one(model, Wi, q0_square, dt, total_time, particles)
            record(model, index, Wi, result, completed)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
            futures = {
                executor.submit(_simulate_one, model, Wi, q0_square, dt, total_time, particles):
                (model, index, Wi)
                for model, index, Wi in jobs
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                model, index, Wi = futures[future]
                record(model, index, Wi, future.result(), completed)

    return time, np.stack(results["Hookean"]), np.stack(results["FENE"])


def _make_axes(rows, columns, title):
    """参考 BrownianMotion_2D_withF，使用紧凑的 constrained_layout 网格。"""
    figsize = (10.0, 7.0) if rows == 2 else (10.0, 3.8)
    fig, axes = plt.subplots(
        rows, columns, figsize=figsize, squeeze=False, constrained_layout=True,
    )
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.04, hspace=0.04)
    fig.suptitle(title, fontsize=15)
    return fig, axes


def _selected_index(Wi_list, selected_Wi):
    matches = np.flatnonzero(np.isclose(Wi_list, selected_Wi, rtol=0, atol=1e-10))
    if len(matches) != 1:
        raise ValueError(f"Wi={selected_Wi:g} must appear exactly once in Wi_list.")
    return int(matches[0])


def plot_hookean_time_curves(time, Wi_list, hookean_q2, dt, particles):
    fig, axes = _make_axes(
        2, 2,
        r"Hookean spring: $\langle Q_x^2\rangle$ versus time",
    )
    for ax, Wi in zip(axes.flat, SELECTED_WI):
        index = _selected_index(Wi_list, Wi)
        numerical = hookean_q2[index, :, 0]
        theoretical = hookean_qx2_analytical(
            time, Wi, bd.BOLTZMANN_CONSTANT, bd.TEMPERATURE,
            bd.SPRING_CONSTANT, bd.GAMMA,
        )
        ax.plot(time, numerical, color="C0", linewidth=2, label="simulation")
        ax.plot(time, theoretical, color="black", linestyle="--", linewidth=2,
                label="exact theory")
        ax.set_title(f"Wi = {Wi:g}")
        ax.set_xlabel("time")
        ax.set_ylabel(r"$\langle Q_x^2\rangle$")
        ax.set_xlim(float(time[0]), float(time[-1]))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
        ax.legend(loc="lower right" if Wi <= 0.5 else "upper left", fontsize=9,
                  frameon=True, facecolor="white", edgecolor="0.85", framealpha=0.97)
    return fig


def plot_fene_time_curves(time, Wi_list, fene_q2, q0_square, dt, particles):
    fig, axes = _make_axes(
        2, 2,
        rf"FENE spring: $\langle Q_x^2\rangle$ versus time ($Q_0^2={q0_square:g}$)",
    )
    for ax, Wi in zip(axes.flat, SELECTED_WI):
        index = _selected_index(Wi_list, Wi)
        ax.plot(time, fene_q2[index, :, 0], color="C0", linewidth=2, label="simulation")
        ax.set_title(f"Wi = {Wi:g}")
        ax.set_xlabel("time")
        ax.set_ylabel(r"$\langle Q_x^2\rangle$")
        ax.set_xlim(float(time[0]), float(time[-1]))
        # 各面板保留完整时间曲线；按该面板的数值量级设置纵轴，避免小 Wi 曲线被压平。
        # 绘图区几何尺寸相同，不要求纵轴范围相同。
        ax.set_ylim(0.0, min(q0_square, max(1.0, float(np.max(fene_q2[index, :, 0]))) * 1.15))
        ax.legend(loc="upper left", fontsize=9, frameon=True,
                  facecolor="white", edgecolor="0.85", framealpha=0.97)
    return fig


def plot_final_moments(time, Wi_list, hookean_q2, fene_q2, q0_square, dt, particles):
    fig, axes = _make_axes(
        1, 2,
        rf"Final-time $\langle Q_x^2\rangle$ versus Wi ($T={time[-1]:g}$)",
    )
    hookean_ax, fene_ax = axes.flat
    hookean_ax.plot(Wi_list, hookean_q2[:, -1, 0], "o-", color="C0",
                    markersize=3.2, linewidth=1.5, label="simulation")
    hookean_ax.set_yscale("log")
    hookean_ax.set_title("Hookean spring (log scale)")
    fene_ax.plot(Wi_list, fene_q2[:, -1, 0], "o-", color="C0",
                 markersize=3.2, linewidth=1.5, label="simulation")
    fene_ax.set_title(rf"FENE spring ($Q_0^2={q0_square:g}$)")
    fene_ax.set_ylim(0.0, q0_square)
    for ax in (hookean_ax, fene_ax):
        ax.axvline(0.5, color="0.45", linestyle=":", linewidth=1.2, label="Wi = 0.5")
        ax.set_xlim(float(Wi_list[0]), float(Wi_list[-1]))
        ax.set_xlabel("Wi")
        ax.set_ylabel(r"$\langle Q_x^2(T)\rangle$")
        ax.legend(loc="upper left", fontsize=9, frameon=True,
                  facecolor="white", edgecolor="0.85", framealpha=0.97)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Calculate the dumbbell moments and save three compact PDF figures.")
    parser.add_argument("--q0-square", type=float, default=DEFAULT_Q0_SQUARE)
    parser.add_argument("--dt", type=float, default=bd.TIME_STEP)
    parser.add_argument("--total-time", type=float, default=bd.TOTAL_TIME)
    parser.add_argument("--particles", type=int, default=bd.PARTICLE_NUMBER)
    parser.add_argument("--workers", type=int, default=4, help="Parallel processes; use 1 for serial execution.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    positive = [args.q0_square, args.dt, args.total_time]
    if not all(np.isfinite(value) and value > 0 for value in positive):
        parser.error("q0-square, dt and total-time must be positive and finite.")
    if args.particles < 1 or args.workers < 1:
        parser.error("particles and workers must be positive integers.")
    if int(round(args.total_time / args.dt)) < 1:
        parser.error("total-time must contain at least one time step.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Wi_list = np.linspace(0.1, 2.0, 39)
    print(f"Output: {args.output_dir.resolve()}", flush=True)
    print(f"Q0^2={args.q0_square:g}, dt={args.dt:g}, T={args.total_time:g}, N={args.particles}", flush=True)
    time, hookean_q2, fene_q2 = run_simulations(
        Wi_list, args.q0_square, args.dt, args.total_time, args.particles, args.workers,
    )
    figures = [
        ("01_Hookean_Qx2_vs_time", plot_hookean_time_curves(
            time, Wi_list, hookean_q2, args.dt, args.particles)),
        ("02_FENE_Qx2_vs_time", plot_fene_time_curves(
            time, Wi_list, fene_q2, args.q0_square, args.dt, args.particles)),
        ("03_Final_Qx2_vs_Wi", plot_final_moments(
            time, Wi_list, hookean_q2, fene_q2, args.q0_square, args.dt, args.particles)),
    ]
    for name, fig in figures:
        fig.savefig(args.output_dir / f"{name}.pdf", format="pdf", facecolor="white",
                    metadata={"Subject": f"dt={args.dt:g}; N={args.particles}; final time={time[-1]:g}; Q0 squared={args.q0_square:g}; seed={bd.RANDOM_SEED}"})
        plt.close(fig)
        print(f"Saved: {name}.pdf", flush=True)
    print("The Hookean analytical curves describe the continuous stochastic equation.")
    print("Finite-step bias and finite-sample fluctuations can produce visible differences.")
    print("Final-time values are not automatically steady-state values (Hookean has no finite steady second moment at Wi >= 0.5).")


if __name__ == "__main__":
    freeze_support()
    main()
