import numpy as np


def obervables(r):

    relative = r - r.mean(axis=1, keepdims=True)           # 珠子相对质心的位置向量

    rg2 = np.mean(np.sum(relative**2, axis=-1), axis=1)

    end_to_end = r[:, -1, :] - r[:, 0, :]
    end2 = np.sum(end_to_end**2, axis=-1)

    span_x = np.ptp(r[:, :, 0], axis=1)                    # x方向伸展量

    return rg2, end2, span_x

def theoretical_relaxation_time(N, H, gamma):
    return gamma / (
        4 * H * np.sin(np.pi / (2 * N))**2
    )

def first_rouse_mode(r):
    N = r.shape[1]    # r.shape = (M, N, 3)

    weights = np.cos(
        np.pi * (np.arange(N) + 0.5) / N
    )

    # 去掉整条链的平移
    relative_position = r - r.mean(axis=1, keepdims=True)
    
    # 第一Rouse模态（上的投影）
    mode = np.mean(
        relative_position * weights[None, :, None],
        axis=1,
    )

    return mode    # shape = (M, 3) 

def mode_correlation(mode_history):
    mode_history = np.asarray(mode_history)    # shape = (t, M, 3) 

    initial_mode = mode_history[0]

    # 每条链当前形变与初始形变的点积
    dot_product = np.sum(
        mode_history * initial_mode,
        axis=-1,
    )

    # 对 M 条链取平均
    correlation = np.mean(dot_product, axis=1)    # shape(t,)

    return correlation / correlation[0]

def fit_relaxation_time(
    time,
    correlation,
    start_time,
    end_time,
):
    time = np.asarray(time)
    correlation = np.asarray(correlation)    # shape(t,)

    selected = (
        (time >= start_time)
        & (time <= end_time)
    )

    slope, intercept = np.polyfit(
        time[selected],
        np.log(correlation[selected]),
        1,
    )

    return -1 / slope

