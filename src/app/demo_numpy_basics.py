import numpy as np
import math
import time
from .physics_utils import (
    MassSpringSystem,
)

# 1. Time vector
# ---------------------------------------------------------------

# start = 0, stop = 10.01 (to include 10), step = 0.01
time_arange = np.arange(0, 10.01, 0.01)

# start = 0, stop = 10, num = 1001 points. num_points = (stop - start) / num + 1
time_linspace = np.linspace(0, 10, 1001)


print("time_linspace:", time_linspace.shape, time_linspace.dtype)

first_five_arange = time_arange[:5]
last_five_arange = time_arange[len(time_arange) - 5:]
first_five_linspace = time_linspace[:5]
last_five_linspace = time_linspace[len(time_linspace) - 5:]

# Print shapes and data types
print(f"time_arange: {time_arange.shape}, {time_arange.dtype}")
print(f"first_five_arange: {first_five_arange}")
print(f"last_five_arange: {last_five_arange}")
print(f"time_linspace: {time_linspace.shape}, {time_linspace.dtype}")
print(f"first_five_linspace: {first_five_linspace}")
print(f"last_five_linspace: {last_five_linspace}")


# 2. Force array (vectorized)
# ------------------------------------------------------------

# force-vs-time array for a simple harmonic input (Vectorized)
def harmonic_force_vectorized(t: np.ndarray, F0: float, omega_n: float, phase: float = 0.0) -> np.ndarray:
    """
    Computes a simple harmonic force array: F(t) = F0 * cos(omega_n * t + phase)
    using vectorization
    """
    return F0 * np.cos(omega_n * t + phase)


# force-vs-time array for a simple harmonic input (pure Python loop)
def harmonic_force_pure(t: list, F0: float, omega_n: float, phase: float = 0.0) -> list:
    """
    Computes a simple harmonic force array: F(t) = F0 * cos(omega_n * t + phase)
    using pure Python loop version
    """
    return [F0 * math.cos(omega_n * x + phase) for x in t]


# instantiate a Mass Spring object
sys = MassSpringSystem(m=1.0, k=4.0, c=0.5)
x = 0.25
v = 0.1
omega_n = sys.omega_n
zeta = sys.zeta
initial_force = sys.force(x, v)


# the performance speed difference
# calculating the time it takes to run pure python loop
start_time_pure = time.perf_counter()
harmonic_force_pure(
    t=time_linspace.tolist(), F0=initial_force, omega_n=omega_n)
end_time_pure = time.perf_counter()

execution_time_pure = end_time_pure - start_time_pure

# calculating the time it takes to run vectorized Numpy version
start_time_vectorized = time.perf_counter()
harmonic_force_vectorized(
    t=time_linspace, F0=initial_force, omega_n=omega_n)
end_time_vectorized = time.perf_counter()

execution_time_vectorized = end_time_vectorized - start_time_vectorized

# Ratio of Vectorized execution time over Pure Python loop execution
speedup = round(execution_time_pure / execution_time_vectorized, 2)
print(speedup)

# 3. Analytical undamped solution (vectorized)
# --------------------------------------------------------
# Full x(t) array in one vectorized expression for undamped system(c=0).
x_t = x * np.cos(omega_n * time_linspace) + \
    (v * np.sin(omega_n * time_linspace) / omega_n)


# 4. Basic inspection & masking
# ---------------------------------------------------------

# Maximum absolute displacement
max_disp = np.max(np.abs(x_t))


# All times when the mass is within 5 % of the origin
threshold = 0.05 * max_disp
mask = np.abs(x_t) < threshold
times_near_origin = time_linspace[mask]

# The approximate period from the zero crossings
sign_change = np.diff(np.sign(x_t))
rising_crossings = np.where(sign_change > 0)[0]   # only positive changes

if len(rising_crossings) >= 2:
    periods = np.diff(time_linspace[rising_crossings])
    approx_period = np.mean(periods)
else:
    approx_period = np.nan
