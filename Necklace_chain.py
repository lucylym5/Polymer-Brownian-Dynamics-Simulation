""" position: r.shape = (M, N, 3)
    M: chain number, N: bead number, the position vector of this bead
    connection vector:  q = r[:, 1:, :] - [:, :-1, :]
    
    N = 2 → Dumbell
""" 
import numpy as np


def connection_vector(
        r: np.ndarray,
):
    q = r[:, 1:, :] - r[:, :-1, :]
    
    return q


def potential_energy(
        r: np.ndarray,
        H: float,
):
    q = connection_vector(r)
    q_square = q**2
    potential_energy = 0.5 * H * (q_square).sum(axis=(2,1))
    
    return potential_energy     # potential_energy.shape=(M,)


def spring_force(
        name: str,
        r: np.ndarray,
        H: float,
        Q0_square: float,
):
    q = connection_vector(r)
    q2 = np.sum(q**2, axis=-1, keepdims=True)

    if name == "Hookean":
        tension = H * q
    elif name == "FENE":
        tension = H * q / (1 - q2 / Q0_square)

    force = np.zeros_like(r)
    force[:, :-1, :] += tension
    force[:, 1:, :] -= tension     # 每根弹簧向+方向拉左珠子，-方向拉右珠子
    
    return force


def initialize_chain(
        name: str,
        M: int, 
        N:int,
        kBT: float,
        H: float, 
        sigma: float,
        rng,
        dimension=3,
        Q0_square=None,
        max_trials=1000,
        max_restarts=100,
):
    min_distance = 2**(1/6) * sigma
    min_distance2 = min_distance**2

    r = np.zeros((M, N, dimension))
    for m in range(M):
        for restart in range(max_restarts):
            r[m]=0
            
            for n in range(1, N):
                for trial in range(max_trials):
                    if name == "Hookean":
                        q = np.sqrt(kBT / H) * rng.standard_normal(dimension)
                    else:
                        direction = rng.standard_normal(dimension)
                        norm = np.linalg.norm(direction)
                        if norm == 0:
                            continue
                        direction /= norm

                        x = rng.beta(
                            dimension / 2,
                            1 + H * Q0_square / (2 * kBT),
                        )
                        q = np.sqrt(Q0_square * x) * direction
                    
                    # 新位置与全部已放置珠子比较
                    candidate = r[m, n-1] + q
                    dr = candidate - r[m, :n]
                    distance2 = np.sum(dr**2, axis=-1)
                    if np.all(distance2 >= min_distance2):
                        r[m, n] = candidate
                        break
                else:
                    # 当前珠子放置失败，重新生成整条链
                    break
            else:
                # 全部珠子放置成功，结束本条链的重试
                break
        else:
            raise RuntimeError(
                f"第 {m} 条链生成失败，请检查参数或增加尝试次数。"
            ) 
        
    r -= r.mean(axis=1, keepdims=True)
    return r


def WCA_force(
        r: np.ndarray,
        epsilon: float,
        sigma: float,
):
    force = np.zeros_like(r, dtype=float)
    if epsilon == 0:
        return force
    
    cutoff = 2**(1/6) * sigma                   # 将LJ势截断，只保留排斥力
    for gap in range(1, r.shape[1]):            # gap 为两珠间隔
        dr =  r[:, gap:, :] - r[:, :-gap, :]    # shape = (M, N-gap, 3)
        distance2 = np.sum(dr**2, axis=-1)      # shape = (N, N-gap)

        active = distance2 < cutoff**2
        factor = np.zeros_like(distance2, dtype=float)
        s6 = (sigma**2 / distance2[active])**3

        factor[active] = (
            24 * epsilon / distance2[active]
            * (2*s6**2 - s6)
        )

        pair_force = factor[..., None] * dr

        force[:, gap:, :] += pair_force
        force[:, :-gap, :] -= pair_force

    return force

