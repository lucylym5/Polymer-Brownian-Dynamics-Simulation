import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ------------------------------ parameters ------------------------------
BOLTZMANN_CONSTANT = 1.0    # k_B
TEMPERATURE = 1.0           # T
GAMMA = 1.0                 # friction coefficient
TIME_STEP = 0.01            # dt
TOTAL_TIME = 10.0
PARTICLE_NUMBER = 5000
INITIAL_POSITION = (0, 0)
RANDOM_SEED = 2026

def cal_D(kB, T, gamma):
    return kB * T / gamma

""" Euler--Maruyama method 模拟2D布朗运动 """
def brownian_2d(
    kB = BOLTZMANN_CONSTANT,
    T = TEMPERATURE,
    gamma = GAMMA,
    dt=TIME_STEP,
    total_time=TOTAL_TIME,
    particle_number=PARTICLE_NUMBER,
    initial_position=INITIAL_POSITION,
    random_seed=RANDOM_SEED,
):
    diffusion_coefficient = cal_D(kB, T, gamma)
    number_of_steps = int(round(total_time / dt))    # round(number, 3) 四舍五入保留3位小数
    time = np.arange(number_of_steps + 1) * dt       # time 数组
    rng = np.random.default_rng(random_seed)

    # 每一维都满足Langevin方程
    displacement = np.sqrt(2 * diffusion_coefficient * dt) * rng.normal(
        size=(particle_number, number_of_steps, 2)
    )

    # 位置矩阵
    position = np.empty((particle_number, number_of_steps + 1, 2))
    position[:, 0, :] = initial_position
    position[:, 1:, :] = np.asarray(initial_position) + np.cumsum(    # 沿axis=1这一维累加
        displacement, axis=1
    )    
    return time, position

""" 一维高斯分布 概率密度函数 """
def gaussian_1d(x, mean, variance):   
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)


""" 2D扩散 粒子径向概率分布 解析解 """
def radial_probability_density(radius, diffusion_coefficient, time):
    return radius / (2.0 * diffusion_coefficient * time) * np.exp(
        -(radius**2) / (4.0 * diffusion_coefficient * time)
    )

""" 绘图 对比模拟和解析结果 """
def plot_results(time, position, diffusion_coefficient):
    initial_position = position[0, 0]
    displacement = position - initial_position
    squared_radius = np.sum(displacement**2, axis=2)     # 粒子位移平方
    simulated_msd = np.mean(squared_radius, axis=0)      # 均方位移
    theoretical_msd = 4.0 * diffusion_coefficient * time 
    fitted_slope, fitted_intercept = np.polyfit(time, simulated_msd, 1)
    theoretical_slope = 4.0 * diffusion_coefficient
    relative_error = abs(fitted_slope - theoretical_slope) / theoretical_slope * 100.0

    print("拟合斜率 =", fitted_slope)
    print("理论斜率 =", theoretical_slope)
    print(f"斜率相对误差 = {relative_error:.3f} %")
    print("拟合截距 =", fitted_intercept)

    final_time = time[-1]
    final_x = position[:, -1, 0]
    final_radius = np.sqrt(squared_radius[:, -1])

    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

# axes(0, 0)：少量代表性粒子
    for particle_index in range(min(10, position.shape[0])):
        axes[0, 0].plot(
            position[particle_index, :, 0],
            position[particle_index, :, 1],
            linewidth=1.0,
            alpha=0.8,       # 透明度
        )
    axes[0, 0].scatter(*initial_position, color="black", marker="o", label="start", zorder=5)
    axes[0, 0].set_title("Sample trajectories")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].xaxis.set_major_locator(MultipleLocator(5))  # x 轴每隔 5 标一次
    axes[0, 0].yaxis.set_major_locator(MultipleLocator(5)) 
    axes[0, 0].set_aspect("equal", adjustable="datalim") 
    axes[0, 0].legend()

# axes(0, 1)：MSD 比较
    axes[0, 1].plot(time, simulated_msd, "o", markersize=2.5, label="simulation")
    axes[0, 1].plot(
        time,
        theoretical_msd,
        "--",
        color='green',
        label=r"theory: $\langle r^2\rangle=4Dt$",
        linewidth=2.5,
    )
    axes[0, 1].plot(
        time,
        fitted_slope * time + fitted_intercept,
        ":",
        color="red",
        label="linear fit",
        linewidth=2,
    )
    axes[0, 1].set_title("Mean-square displacement")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel(r"$\langle |\mathbf{r}(t)-\mathbf{r}(0)|^2\rangle$")
    axes[0, 1].legend()

# 绘制检验 x 是否满足高斯分布
    x_grid = np.linspace(final_x.min(), final_x.max(), 500)
    x_variance = 2.0 * diffusion_coefficient * final_time
    axes[1, 0].hist(
        final_x, bins=55, density=True, alpha=0.55, label="simulation"
    )
    axes[1, 0].plot(
        x_grid,
        gaussian_1d(x_grid, initial_position[0], x_variance),
        "--",
        linewidth=2,
        label=r"theory: $N(x_0,2Dt)$",
    )
    axes[1, 0].set_title(f"x distribution at t = {final_time:g}")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("probability density")
    axes[1, 0].legend()

# 绘制 r 的分布
    radius_grid = np.linspace(0.0, final_radius.max(), 500)
    axes[1, 1].hist(
        final_radius, bins=55, density=True, alpha=0.55, label="simulation"
    )
    axes[1, 1].plot(
        radius_grid,
        radial_probability_density(radius_grid, diffusion_coefficient, final_time),
        "--",
        linewidth=2,
        label="theory (Rayleigh distribution)",
    )
    axes[1, 1].set_title(f"Radial distribution at t = {final_time:g}")
    axes[1, 1].set_xlabel(r"$r=|\mathbf{r}(t)-\mathbf{r}(0)|$")
    axes[1, 1].set_ylabel("probability density")
    axes[1, 1].legend()

    figure.suptitle("2D free Brownian motion", fontsize=15)
    figure.savefig("BrownianMotion_2D_free.pdf")


def main():
    time, position = brownian_2d()
    plot_results(time, position, cal_D(BOLTZMANN_CONSTANT, TEMPERATURE, GAMMA))


if __name__ == "__main__":
    main()
