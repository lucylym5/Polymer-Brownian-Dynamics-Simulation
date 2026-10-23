import numpy as np


def obervables(r):

    relative = r - r.mean(axis=1, keepdims=True)           # 珠子相对质心的位置向量

    rg2 = np.mean(np.sum(relative**2, axis=-1), axis=1)    # 回转半径的平方

    end_to_end = r[:, -1, :] - r[:, 0, :]                  # 末端距
    end2 = np.sum(end_to_end**2, axis=-1)

    span_x = np.ptp(r[:, :, 0], axis=1)                    # x方向伸展量

    return rg2, end2, span_x

