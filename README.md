# Zimm/Rouse chain Elongation Simulation
高分子流场操控仿真环境搭建

## noFlow：无流动胡克链与标度图

在包含 NumPy 和 Matplotlib 的 Python 环境中运行：

~~~powershell
python NoFlow.py
~~~

默认 M=500、dt=0.05，扫描 N=20,30,...,100，H=gamma=kBT=1，随机种子为2026。
每个 N 从胡克平衡构象出发，复用 euler_maruyama_step()，推进约3个理论最长相关时间。
总时间随N增大：N=20时约121.8，N=100时约3039.9；默认最多同时计算4个N。
本脚本使用自己的配置，不使用 parameters.py 中的 TIME_STEP、MOLECULE_NUMBER 和 TOTAL_TIME。

~~~powershell
python NoFlow.py --chains 500 --dt 0.05 --duration 3
python NoFlow.py --n-step 1
python NoFlow.py --fit-start 0.3 --fit-end 1.2 --show
~~~

--duration 表示总时间与理论相关时间之比；--fit-start、--fit-end 也以理论相关时间为单位。
--n-step 1 逐个扫描所有整数N；--show 在保存后打开图片窗口。
当前结果和默认输出目录均为 noFlow_results。

### 绘图与拟合

同一个figure包含两个子图，采用等宽布局和 set_box_aspect(3/4)，
保证两块绘图区等大，宽高比均为4:3。左上角标注拟合斜率与R²，不标注Expected slope。

- 左图：log(sqrt(<Rg^2>))-log(N)。每条链先相对于自己的质心计算Rg²，
  再对链及记录时刻平均、最后开平方。
- 右图：log(tau_R)-log(N)。tau_R从第一Rouse模态的相关函数拟合得到。
- 两图均先取自然对数，再用np.polyfit拟合直线，并在原图绘制拟合线。

默认拟合0.2*tau_theory到1.5*tau_theory的相关函数。
理论值只用于安排运行长度、采样、拟合窗口及作为对照，不替代模拟测量值。
若某条曲线在整个选定窗口中无法给出有效正衰减时间，该点会被省略，不会替换为理论时间。
这里tau_R是模态向量的相关时间，不是用于二阶矩或Wi的tau_R/2。

每个理论相关时间记录约100帧，以避免在长链上保存每一步的所有模态数据。
输出中的时间仍为真实记录时刻，没有用帧编号代替物理时间。

### 输出文件

- scaling_laws.png：两个等大长方形绘图区的标度图。
- mode_correlations.png：以t/tau_theory为横轴的相关函数，灰色区域为拟合窗口。
- scaling_results.csv：各N的尺寸、相关时间、实际总时间、拟合窗口和采样信息。
- mode_correlations.npz：实际记录时间与相关曲线。
- summary.json：运行参数、拟合结果及两块绘图区的实际尺寸。

## 当前长时间运行结果

N=20,30,...,100，M=500，dt=0.05，每个N推进约3个理论相关时间：

| 量 | 拟合斜率 | 回归标准误差 | R² |
|---|---|---|---|
| Rg = sqrt(<Rg²>) | 0.50774 | 0.00256 | 0.99982 |
| 第一模态相关时间tau_R | 2.06301 | 0.06183 | 0.99375 |

数据和图片均保存在noFlow_results。回归标准误差描述不同N的拟合散布；
每个N的相关曲线、拟合区间和实际运行时间均可在保存的数据中查阅。
