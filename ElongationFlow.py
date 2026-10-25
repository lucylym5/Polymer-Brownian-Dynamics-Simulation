"""
    计算平面拉伸流中, 链伸展量 x/l_c 随时间 t 变化曲线
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib

import Necklace_chain as nc
import observables as ob
import parameters as p
import time_stepping as ts

def fit_power_law(N, values):
    """对 log(values)-log(N) 作直线拟合。"""
    x = np.log(np.asarray(N, dtype=float))
    y = np.log(np.asarray(values, dtype=float))
    coefficient, covariance = np.polyfit(x, y, 1, cov=True)
    residual = y - np.polyval(coefficient, x)
    return {
        "slope": float(coefficient[0]),
        "intercept": float(coefficient[1]),
        "slope_se": float(np.sqrt(covariance[0, 0])),
        "r_squared": float(1 - np.sum(residual**2) / np.sum((y - y.mean())**2)),
    }