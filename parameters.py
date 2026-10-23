import numpy as np


#   physical constants
KB = 1.380649e-23

#   dimensionless params
KB_DIMLESS = 1
TEMP_DIMLESS = 1
GAMMA = 1
TIME_STEP = 5e-3
TOTAL_TIME = 10
MOLECULE_NUMBER = 100
INITIAL_POSITION = (0, 0)
RANDOM_SEED = 2026

# 流场
def generate_velocity_gradient(strain_rate):
    return np.array([
        [strain_rate, 0.0, 0.0],
        [0.0, -strain_rate, 0.0],
        [0.0, 0.0, 0.0]
    ])