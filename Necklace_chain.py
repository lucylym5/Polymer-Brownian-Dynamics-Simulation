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
        M: int, 
        N:int,
        kBT: float,
        H: float, 
        rng, 
        dimension=3
):
    q = np.sqrt(kBT / H) * rng.standard_normal(size=(M, N - 1, dimension))    # 构象初始化
    r = np.zeros((M, N, dimension))                                           # 位置初始化
    r[:, 1:, :] = np.cumsum(q, axis=1)
    r -= r.mean(axis=1, keepdims=True)

    return r

