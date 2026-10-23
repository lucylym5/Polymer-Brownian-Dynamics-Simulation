import numpy as np

rng = np.random.default_rng(2026)

# 初始参数和位置
kBT = 1.0
zeta = 1.0      # 摩擦系数
D = kBT / zeta  # 扩散系数

dt = 0.01
x = 0.0
total_time = 10


# M 次实验，检查扩散规律
M = 10000
n_steps = int(round(total_time / dt))

x = np.zeros(M)
times = np.arange(n_steps + 1) * dt   # 记录 t
msd = np.zeros(n_steps + 1)           # 均方位移-t

for n in range(n_steps):
    Force = 0                   # 外力
    z = rng.standard_normal(M)
    dx = Force / zeta * dt + np.sqrt(2 * D * dt) * z
    x = x + dx
    
    msd[n + 1] = np.mean(x**2)

a, b = np.polyfit(times, msd, 1)
a_theory = 2 * D
error_percent = abs(a - a_theory) / a_theory * 100

print("拟合斜率 =", a)
print("理论斜率 =", a_theory)
print(f"斜率相对误差 = {error_percent} %")
print("拟合截距 =", b)

# 画图
import matplotlib.pyplot as plt

plt.plot(times, msd, "o", markersize=2.5, label="Simulation")
plt.plot(times, times * 2 * D, 
         linestyle="-", color="green", linewidth=2.5,
         label=r"Theory: $\langle x^2\rangle=2Dt$")
plt.plot(times, a * times + b,
         ":", color="red", linewidth=2.5,
         label="Linear fit")
plt.xlabel("Time")
plt.ylabel("MSD")

plt.xlim(0, 10)

plt.gca().set_box_aspect(0.9)
plt.legend()
# plt.savefig("msd1.pdf")

