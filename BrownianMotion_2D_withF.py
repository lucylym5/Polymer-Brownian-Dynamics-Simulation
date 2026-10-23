import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ------------------------------ parameters ------------------------------
BOLTZMANN_CONSTANT = 1.0    # k_B
TEMPERATURE = 1.0           # T
GAMMA = 1.0                 # friction coefficient
SPRING_CONSTANT = 1.0       # k; force F = - kr
TIME_STEP = 0.005           # dt
TOTAL_TIME = 10
PARTICLE_NUMBER = 5000
INITIAL_POSITION = (0.0, 0.0)
RANDOM_SEED = 2026


def brownian_2d_with_force(
    kB = BOLTZMANN_CONSTANT,
    T = TEMPERATURE,
    gamma = GAMMA,
    k_spring = SPRING_CONSTANT,
    dt = TIME_STEP,
    total_time = TOTAL_TIME,
    particle_number = PARTICLE_NUMBER,
    initial_position = INITIAL_POSITION,
    random_seed = RANDOM_SEED,
):
    step_number = int(round(total_time / dt))
    time = np.arange(step_number + 1) * dt
    rng = np.random.default_rng(random_seed)

    D = (                  # 扩散系数
        kB * T / gamma
    )
    relaxation_rate = k_spring / gamma
    noise = np.sqrt(2.0 * D * dt)

    position = np.empty((particle_number, step_number + 1, 2))
    position[:, 0, :] = initial_position

    # Euler--Maruyama method
    for step in range(step_number):
        deterministic_increment = -relaxation_rate * position[:, step, :] * dt
        stochastic_increment = noise * rng.normal(
            size=(particle_number, 2)
        )
        position[:, step + 1, :] = (
            position[:, step, :] + deterministic_increment + stochastic_increment
        )

    return time, position

"""平均位置、均方坐标、均方半径"""
def theoretical_displacement(
    time,
    initial_position,
    kB = BOLTZMANN_CONSTANT,
    T = TEMPERATURE,
    gamma = GAMMA,
    k_spring = SPRING_CONSTANT,
):
    relaxation_rate = k_spring / gamma
    decay = np.exp(-relaxation_rate * time)
    mean = np.asarray(initial_position)[:, None] * decay[None, :]
    coordinate_variance = (
        kB * T
        / k_spring
        * (1.0 - np.exp(-2.0 * relaxation_rate * time))
    )
    mean_squared_radius = np.sum(mean**2, axis=0) + 2.0 * coordinate_variance
    return mean, coordinate_variance, mean_squared_radius


"""一维高斯分布概率函数"""
def gaussian_1d(x, mean, variance):
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(
        2.0 * np.pi * variance
    )


def plot_results(
    time,
    position,
    kB = BOLTZMANN_CONSTANT,
    T = TEMPERATURE,
    gamma = GAMMA,
    k_spring = SPRING_CONSTANT,
):
    """Plot the simulation and the exact Ornstein--Uhlenbeck solution."""
    initial_position = position[0, 0]
    theoretical_mean, theoretical_variance, theoretical_second_moment = (
        theoretical_displacement(
            time,
            initial_position,
            kB,
            T,
            gamma,
            k_spring,
        )
    )
    simulated_mean = np.mean(position, axis=0)
    simulated_variance = np.mean(
        (position - simulated_mean[None, :, :]) ** 2, axis=0
    )
    simulated_second_moment = np.mean(np.sum(position**2, axis=2), axis=0)
    final_relative_error = (
        abs(simulated_second_moment[-1] - theoretical_second_moment[-1])
        / theoretical_second_moment[-1]
        * 100.0
    )

    print("Final simulated <x^2+y^2> =", simulated_second_moment[-1])
    print("Final theoretical <x^2+y^2> =", theoretical_second_moment[-1])
    print(f"Final relative error = {final_relative_error:.3f} %")

    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    # 图1：等势线和少量粒子轨迹
    box = axes[0, 0].get_window_extent()
    ratio = box.width / box.height

    limit = max(3.5, 1.25 * np.max(np.abs(position[:10])))
    grid = np.linspace(-limit, limit, 150)                    # 等势线绘制区域范围
    xx, yy = np.meshgrid(grid*ratio, grid)
    potential = 0.5 * k_spring * (xx**2 + yy**2)
    axes[0, 0].contour(xx, yy, potential, levels=10, cmap="Greys", alpha=0.5)    # 等势线
    for particle_index in range(min(10, position.shape[0])):                     # 粒子样本轨迹
        axes[0, 0].plot(
            position[particle_index, :, 0],
            position[particle_index, :, 1],
            linewidth=1.0,
            alpha=0.8,
        )
    axes[0, 0].scatter(*initial_position, color=("black"), label="start", zorder=5)
    axes[0, 0].scatter(0.0, 0.0, color="red", marker="x", label="trap center")
    axes[0, 0].set_title(r"Sample trajectories in $U(r)=kr^2/2$")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].set_aspect("equal", adjustable="datalim")    # 控制 x,y 长度比例一致
    axes[0, 0].legend()


    axes[0, 1].plot(time, simulated_variance[:, 0], label=r"simulation: $\langle x^2 \rangle$")
    axes[0, 1].plot(time, simulated_variance[:, 1], label=r"simulation: $\langle y^2 \rangle$")
    axes[0, 1].plot(
        time,
        theoretical_variance,
        "--",
        color="black",
        linewidth=2,
        label=r"theory: $\frac{k_BT}{k}(1-e^{-2kt/\gamma})$",
    )
    axes[0, 1].set_title("Variance of each coordinate")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("variance")
    axes[0, 1].legend(fontsize=9)


    axes[1, 0].plot(
        time, simulated_second_moment, label="simulation", linewidth=2
    )
    axes[1, 0].plot(
        time,
        theoretical_second_moment,
        "--",
        color = "black",
        label=r"exact theory for $\langle x^2+y^2\rangle$",
        linewidth=2,
    )
    axes[1, 0].axhline(
        2.0 * kB * T / k_spring,
        color="grey",
        linestyle=":",
        label=r"equilibrium: $2k_BT/k$",
    )
    axes[1, 0].set_title("Second moment")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel(r"$\langle x^2+y^2\rangle$")
    axes[1, 0].legend()


    final_x = position[:, -1, 0]
    final_mean_x = theoretical_mean[0, -1]
    final_variance = theoretical_variance[-1]
    x_grid = np.linspace(final_x.min(), final_x.max(), 500)
    axes[1, 1].hist(
        final_x, bins=55, density=True, alpha=0.55, label="simulation"
    )
    axes[1, 1].plot(
        x_grid,
        gaussian_1d(x_grid, final_mean_x, final_variance),
        "--",
        linewidth=2,
        label="exact finite-time theory",
    )
    axes[1, 1].set_title(f"x distribution at t = {time[-1]:g}")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("probability density")
    axes[1, 1].legend()

    figure.suptitle("2D Brownian motion in a harmonic trap", fontsize=15)
    figure.savefig("BrownianMotion_2D_withF.pdf")


def main():
    time, position = brownian_2d_with_force()
    plot_results(time, position)


if __name__ == "__main__":
    main()
