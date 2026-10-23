import numpy as np

rng = np.random.default_rng(2026)

# 初始参数和位置
kBT = 1.0
zeta = 1.0      # 摩擦系数
D = kBT / zeta  # 扩散系数
k = 1           # 刚度系数

dt = 0.005
x = 0.0
total_time = 10


# M 次实验，检查扩散规律
M = 10000
n_steps = int(round(total_time / dt))

x = np.zeros(M)
times = np.arange(n_steps + 1) * dt   # 记录 t
msd = np.zeros(n_steps + 1)           # 均方位移-t

for n in range(n_steps):
    Force = - k * x                   # 外力
    z = rng.standard_normal(M)
    dx = Force / zeta * dt + np.sqrt(2 * D * dt) * z
    x = x + dx
    
    msd[n + 1] = np.mean(x**2)

# 画图
import matplotlib.pyplot as plt

plt.plot(times, msd, "o", markersize=2.5, label="Simulation")
plt.axhline(y=kBT / k,
            linestyle="-", color="green", linewidth=2.5,
            label=r"$k_B T/k$")
# plt.plot(times, a * times + b, ":", color="red", label="Linear fit")
plt.xlabel("Time")
plt.ylabel("MSD")

plt.xlim(0, 10)

plt.gca().set_box_aspect(0.9)
plt.legend()
plt.savefig("msd1'.pdf")

