import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ------------------------------ parameters ------------------------------
BOLTZMANN_CONSTANT = 1.0    # k_B
TEMPERATURE = 1.0           # T
GAMMA = 1.0                 # friction coefficient
SPRING_CONSTANT = 1.0       # H, U = 0.5*H*Q^2
TIME_STEP = 0.005           # dt
TOTAL_TIME = 10
PARTICLE_NUMBER = 5000
INITIAL_POSITION = (0.0, 0.0)
RANDOM_SEED = 2026

def generate_velocity_gradient(strain_rate):
    return np.array([
        [strain_rate, 0.0, 0.0],
        [0.0, -strain_rate, 0.0],
        [0.0, 0.0, 0.0]
    ])

"""初始化高分子构象""" 
def initialize_conformation(
        name: str,
        Q0_square: float,
        kB: float,
        T: float,
        H: float,
        particle_number: int,
        dimension: int,
        rng,
):
    # P(Q) ∝ exp(-HQ^2/2β), N(0, β/H)
    if name == "Hookean":
        standard_deviation = np.sqrt(kB * T/H)
        Q = standard_deviation * rng.standard_normal(size=(particle_number, dimension))
    # Beta 函数直接采样
    elif name == "FENE":    
        Q0 = np.sqrt(Q0_square)
        b = H * Q0_square / kB / T
        # Q²/Q0² 符合beta分布
        normalized_square_length = rng.beta(
            a = dimension / 2,
            b = b/2 + 1,
            size = particle_number,
        )
        # 方向在球面上随机分布
        direction = rng.standard_normal(        # 各分量~N(0,1); p(x, y) ∝ exp[(x^2+y^2)/2]
            size=(particle_number, dimension)
        )

        direction /= np.linalg.norm(            # np.linalg.norm计算direction向量的长度
            direction,
            axis=1,
            keepdims=True,
        )

        Q = (
            Q0
            * np.sqrt(normalized_square_length)[:, None]    # [:, None]: (3,)->(3,1)
            * direction
        )
    return Q


""""单步Euler-Maruyama方法"""
def euler_maruyama_step(
        name: str,
        Q0_square: float,
        Q: np.ndarray ,
        velocity_gradient: np.ndarray,   # 速度梯度张量
        D: float,                        # 扩散系数
        H: float,
        gamma: float,
        dt: float,
        rng,
):
    flow_term = Q @ velocity_gradient.T * dt    # u = KQ
    spring_term = -2 * H * Q / gamma * dt
    noise = np.sqrt(4 * D * dt) * rng.standard_normal(size = Q.shape)

    if name == "Hookean":
        Qnew = Q + flow_term + spring_term + noise
    elif name == "FENE":
        R = Q + flow_term + noise
        Qnew = implicit_FENE(R, Q0_square, H, gamma, dt)

    return Qnew


""""FENE: 半隐式更新Q(t+1), 防止越界"""
def implicit_FENE(
        R: np.ndarray,
        Q0_square: float,
        H: float,
        gamma: float,
        dt: float,
        max_iterations = 60,
):
    # 输入检查
    if Q0_square <= 0:
        raise ValueError("Q0_square must be positive.")

    if H <= 0 or gamma <= 0 or dt <= 0:
        raise ValueError("H, gamma and dt must be positive.")

    Q0 = np.sqrt(Q0_square)

    # 向量 R = Qn + KQn dt + sqrt(4Ddt) zn, y = |Qn+1|/Q0
    R_length = np.linalg.norm(R, axis=1, keepdims=True)    # 计算每行向量长度
    s = R_length / Q0
    a = 2* H * dt / gamma

    # 在 y ∈ (0,1) 二分法解方程
    lower = np.zeros_like(s)
    upper = np.full_like(s, 1 - 1e-12)
    upper_value = (
        upper * (1.0 + a / (1.0 - upper**2)) - s
    )
    if np.any(upper_value < 0.0):
        raise FloatingPointError(
            "FENE implicit solve is too close to Q0. "
            "Reduce time step dt."
        )
    for _ in range(max_iterations):
        middle = 0.5 * (lower + upper)
        middle_fvalue = (
            middle * (1.0 + a / (1.0 - middle**2)) - s
        )
        move_right = middle_fvalue < 0                 # 布尔数组，返回值为每个粒子中点函数值是否<0
        lower = np.where(move_right, middle, lower)    # True，返回第2个参数；False，返回第3个参数
        upper = np.where(move_right, upper, middle)

    y = 0.5 * (lower + upper)
    direction = np.divide(     # 向量除法得到方向
        R,
        R_length,
        out = np.zeros_like(R),
        where = R_length > 0.0,
    )
    Q_new = Q0 * y * direction

    return Q_new


"""在应变率为ε的流场中，一次完整模拟"""
def dumbell_elongation(
        name: str,
        Q0_square: float,
        kB: float,
        T: float,
        gamma: float,                     # 摩擦系数
        H: float, 
        Wi: float,                        # 对应应变率
        dt: float,       
        total_time: float,
        particle_number: int,
        dimension: int,
        random_seed: int,
):
    # 参数准备
    step_number = int(round(total_time/dt))
    time = np.arange(step_number+1) * dt
    rng = np.random.default_rng(random_seed)
    D = (
        kB * T / gamma
    )
    relaxation_time = gamma / (4 * H)
    strain_rate = Wi / relaxation_time
    velocity_gradient = generate_velocity_gradient(strain_rate)

    Q = initialize_conformation(
        name, Q0_square, kB, T, H, particle_number, dimension, rng
    )
    Q2_all = np.zeros((step_number+1, dimension))     # Q的各分量均方
    Q2_all[0] = np.mean(
        Q**2,
        axis=0
    )

    for i in range(1, step_number+1):
        Qnew = euler_maruyama_step(
            name, Q0_square, Q, velocity_gradient, D, H, gamma, dt, rng
        )
        Q2_all[i] =  np.mean(
            Qnew**2,
            axis=0,
        )                        # shape: (step_number+1, 3)
        Q = Qnew
    return time, Q2_all     

"""扫描Wi, 计算Qx-Wi"""
def test(name, Q0_square):
    Wi_list = np.linspace(0.1, 2.0, 39)
    Q2_eq = np.empty(Wi_list.size)

    for index, Wi in enumerate(Wi_list):
        time, Q2_all = dumbell_elongation(
            name,
            Q0_square,
            kB = BOLTZMANN_CONSTANT,
            T = TEMPERATURE,
            gamma = GAMMA,
            H = SPRING_CONSTANT,
            Wi = Wi,
            dt = TIME_STEP,       
            total_time = TOTAL_TIME,
            particle_number = PARTICLE_NUMBER,
            dimension = 3,
            random_seed = RANDOM_SEED,
        )
        Q2_eq[index] = Q2_all[-1, 0]
    return Wi_list, Q2_eq
