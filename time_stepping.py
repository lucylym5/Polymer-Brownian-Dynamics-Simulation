import numpy as np
import Necklace_chain as nc


def euler_maruyama_step(
        name: str,
        r: np.ndarray,
        velocity_gradient: np.ndarray,
        kBT: float,
        gamma: float,
        H: float,
        Q0_square: float,
        dt: float,
        rng,
):
    force = nc.spring_force(
        name, r, H, Q0_square
    )
    flow_term = r @ velocity_gradient.T * dt
    spring_term = force / gamma * dt
    D = kBT / gamma
    noise = np.sqrt(2 * D * dt) * rng.standard_normal(size = r.shape)

    # 考虑每个珠子的 Langevin 方程
    return r + flow_term + spring_term + noise

def semi_implicit_fene_step(
        r: np.ndarray,
        velocity_gradient: np.ndarray,
        kBT: float,
        gamma: float,
        H: float,
        Q0_square: float,
        dt: float,
        rng,
        max_iterations=500,
        tol=1e-9,
):
    Q0 = np.sqrt(Q0_square)
    D = kBT / gamma

    flow_term = r @ velocity_gradient.T * dt
    noise = np.sqrt(2 * D * dt) * rng.standard_normal(size = r.shape)
    r_mid = r + flow_term + noise    # 只考虑流动和随机力时的位置

    # 正确新位置与 r_mid 残差
    def equation_residual(r_new):
        force = nc.spring_force(
            "FENE", r_new, H, Q0_square
        )

        return (
            r_new
            - r_mid
            - force * dt / gamma
        )

    # 从旧位置开始猜测新位置
    candidate_position = r.copy()
    for _ in range(max_iterations):
        residual = equation_residual(candidate_position)
        residual_norm = np.linalg.norm(residual)    # 残差的模长

        if np.max(np.abs(residual)) < tol:    # 残差符合要求时，输出
            return candidate_position

        correction=1
        for _ in range(50):
            trial_position = (
                candidate_position
                - correction * residual
            )
            trial_q = nc.connection_vector(trial_position)
            trial_q_square = np.sum(trial_q**2, axis=-1)
            if np.all(trial_q_square < Q0_square):    # 判断弹簧伸长是否越界
                trial_residual = equation_residual(trial_position)
                if np.linalg.norm(trial_residual) < residual_norm:    # 修正后残差减小，接受这次修正
                    candidate_position = trial_position
                    break
            correction *= 0.5    # 越界或残差未减小，减小修正幅度   
        else:
            raise RuntimeError("位置修正失败，建议减小时间步 dt。")  
    raise RuntimeError(
        "FENE 隐式求解未收敛，建议减小 dt 或增加最大循环次数。"
    )