import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Callable
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar

# Exercise 1 – Basic free response


def free_response_ode(t: float, y: np.ndarray, m: float, c: float, k: float) -> np.ndarray:
    """Mass-spring-damper free response (F = 0).

    State vector y = [x, v]
    """
    x, v = y
    dxdt = v
    dvdt = (-c * v - k * x) / m
    return np.array([dxdt, dvdt])


def forced_ode(
    t: float,
    y: ArrayLike,
    m: float,
    c: float,
    k: float,
    F: Callable[[float], float],
) -> NDArray[np.float64]:
    """Mass-spring-damper external force present F(t).

        State vector y = [x, v]
        """
    x, v = y
    dxdt = v
    dvdt = (F(t) - c*v - k*x) / m
    return np.array([dxdt, dvdt])


def analytical_underdamped(t: np.ndarray, m: float, c: float, k: float,
                           x0: float, v0: float) -> np.ndarray:
    """Exact under-damped free response."""
    omega_n = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(k * m))
    omega_d = omega_n * np.sqrt(1 - zeta**2)

    exp_term = np.exp(-zeta * omega_n * t)
    x = exp_term * (
        x0 * np.cos(omega_d * t)
        + (v0 + zeta * omega_n * x0) / omega_d * np.sin(omega_d * t)
    )
    return x


# Parameters (keep them consistent)
m = 1.0             # ζ = 0.25 → under-damped
c = 0.5
k = 4.0
x0 = 1.0
v0 = 0.0
t_span = (0.0, 15.0)
t_eval = np.linspace(0.0, 15.0, 300)

# Numerical solution
sol = solve_ivp(
    free_response_ode,
    t_span,
    y0=[x0, v0],
    t_eval=t_eval,
    args=(m, c, k),
    rtol=1e-6,
    atol=1e-8,
)

# print("Success:", sol.success)
# print("t shape:", sol.t.shape, "y shape:", sol.y.shape)

# Analytical solution on the same grid
x_analytical = analytical_underdamped(sol.t, m, c, k, x0, v0)

# Quick error check
max_abs_error = np.max(np.abs(sol.y[0] - x_analytical))
# print(f"Max |numerical - analytical| = {max_abs_error:.2e}")

# Plot
# plt.plot(sol.t, sol.y[0], 'b-', label="solve_ivp")
# plt.plot(sol.t, x_analytical, 'r--', label="analytical")
# plt.xlabel("t")
# plt.ylabel("x(t)")
# plt.legend()
# plt.grid(True)
# plt.show()


# Exercise 2 – Different damping regimes

# Relevant parameters
c_under = 0.5  # c = 0.5  → ζ = 0.25
c_critical = 4.0   # c = 4.0  → ζ = 1.0
c_over = 6.0   # c = 6.0  → ζ = 1.5

# solutions
sol_under = solve_ivp(
    free_response_ode,
    t_span,
    y0=[x0, v0],
    t_eval=t_eval,
    args=(m, c_under, k),
    rtol=1e-6,
    atol=1e-8,
)

sol_critical = solve_ivp(
    free_response_ode,
    t_span,
    y0=[x0, v0],
    t_eval=t_eval,
    args=(m, c_critical, k),
    rtol=1e-6,
    atol=1e-8,
)

sol_over = solve_ivp(
    free_response_ode,
    t_span,
    y0=[x0, v0],
    t_eval=t_eval,
    args=(m, c_over, k),
    rtol=1e-6,
    atol=1e-8,
)


# saving to the dedicated file
np.savez("demo_scipy_ode.npz", time_under=sol_under.t,
         displacement_under=sol_under.y[0], velocity_under=sol_under.y[1], time_critical=sol_critical.t, displacement_critical=sol_critical.y[0], velocity_critical=sol_critical.y[1], time_over=sol_over.t, displacement_over=sol_over.y[0], velocity_over=sol_over.y[1])


# load from file and assign to variables
data = np.load("demo_scipy_ode.npz")
t_under = data["time_under"]
x_under = data["displacement_under"]
v_under = data["velocity_under"]

t_critical = data["time_critical"]
x_critical = data["displacement_critical"]
v_critical = data["velocity_critical"]

t_over = data["time_over"]
x_over = data["displacement_over"]
v_over = data["velocity_over"]

# print
# print("under damped mode")
# print("-------------------------")
# print(f"final time: {t_under[-1]}")
# print(f"final displacement: {x_under[-1]}")
# print(f"final velocity: {v_under[-1]}")
# print(f"maximum absolute displacement: {np.max(np.abs(x_under))}")
# print()
# print("critically damped mode")
# print("-------------------------")
# print(f"final displacement: {x_critical[-1]}")
# print(f"final velocity: {v_critical[-1]}")
# print(f"maximum absolute displacement: {np.max(np.abs(x_critical))}")
# print("over damped mode")
# print("-------------------------")
# print(f"final displacement: {x_over[-1]}")
# print(f"final velocity: {v_over[-1]}")
# print(f"maximum absolute displacement: {np.max(np.abs(x_over))}")

# plot
# plt.plot(t_under, x_under, 'b-', label="Under-damped")
# plt.plot(t_critical, x_critical, 'y-', label="Critically-damped")
# plt.plot(t_over, x_over, 'r-', label="Over-damped")
# plt.xlabel("t")
# plt.ylabel("x(t)")
# plt.legend()
# plt.grid(True)
# plt.show()


# Exercise 3 – Dense output & interpolation

# Run with dense output (let the solver choose its own steps)
sol_dense = solve_ivp(
    free_response_ode,
    t_span,
    y0=[x0, v0],
    args=(m, c_under, k),
    dense_output=True,
    rtol=1e-6,
    atol=1e-8,
)

# Fine grid of our choice
t_fine = np.linspace(t_span[0], t_span[1], 1001)

# Interpolate
y_fine = sol_dense.sol(t_fine)   # shape (2, 1001)
x_dense = y_fine[0]

# Reference (analytical or another solve_ivp with t_eval=t_fine)
x_ref = analytical_underdamped(t_fine, m, c_under, k, x0, v0)

max_err = np.max(np.abs(x_dense - x_ref))
# print(f"Max |dense - reference| = {max_err:.2e}")

# plt.plot(t_fine, x_dense, 'b-', label="X dense")
# plt.plot(t_fine, x_ref, 'r-', label="X reference")
# plt.xlabel("t")
# plt.ylabel("x(t)")
# plt.legend()
# plt.grid(True)
# plt.show()


# Exercise 4 – First performance metrics
def peak_time(t: np.ndarray, x: np.ndarray) -> float:
    """Time at which |x| reaches its maximum."""
    # find index for max absolute displacement
    max_index = np.argmax(np.abs(x))
    # find and return corresponding time
    return t[max_index]


def overshoot(x: np.ndarray, x_final: float = 0.0) -> float:
    """Maximum absolute deviation from the final value."""
    return np.max(np.abs(x - x_final))


def settling_time(t: np.ndarray, x: np.ndarray,
                  band: float = 0.05, x_final: float = 0.0) -> float:
    """
    First time after which |x - x_final| remains inside
    ±band * peak_absolute_displacement.
    Returns np.nan if the signal never stays inside the band.
    """
    peak = np.max(np.abs(x - x_final))
    if peak == 0.0:
        return t[0]

    threshold = band * peak
    outside = np.abs(x - x_final) > threshold

    if not np.any(outside):
        return t[0]          # already inside from the start

    last_outside_idx = np.where(outside)[0][-1]

    if last_outside_idx == len(t) - 1:
        return np.nan        # still outside at the final time

    return t[last_outside_idx + 1]


# t_peak = peak_time(t_fine, x_dense)
# print(f"Peak time for reaching absolute maximum: {t_peak}sec")
# abs_overshoot = overshoot(x_dense)
# print(f"Absolute overshoot: {abs_overshoot}")
# t_settling = settling_time(t_fine, x_dense)
# print(
#     f"time it takes to reach and remain within ±5% of the final steady-state value: {t_settling}sec")
# t_peak_critical = peak_time(t_critical, x_critical)
# print(
#     f"Peak time for reaching absolute maximum (critically damped): {t_peak_critical}sec")
# abs_overshoot_critical = overshoot(x_critical, x_critical[-1])
# print(f"Absolute overshoot (critically damped): {abs_overshoot_critical}")
# t_settling_critical = settling_time(
#     t_critical, x_critical, 0.05, x_critical[-1])
# print(
#     f"time it takes to reach and remain within ±5% of the final steady-state value (critically damped): {t_settling_critical}sec")


# Useful forcing functions
def step_force(
    t: float | ArrayLike,
    F0: float = 1.0,
    t_step: float = 0.0,
) -> float | NDArray[np.float64]:
    """Constant force that switches on at t_step.

    Works with both scalar t (for the ODE solver) and array t (for plotting).
    """
    return np.where(t >= t_step, F0, 0.0)


def harmonic_force(
    t: float | ArrayLike,
    F0: float = 1.0,
    omega: float = 2.0,
    phase: float = 0.0,
) -> float | NDArray[np.float64]:
    """Harmonic forcing F(t) = F0 * cos(ωt + φ)."""
    return F0 * np.cos(omega * t + phase)


# Exercise 5 – Forced response

# parameters
m_f = 1.0
c_f = 0.5
k_f = 4.0
F0 = 1.0
t0 = 0.0
tn = 20.0
x0_f = 0.0
v0_f = 0.0
t_span_f = (0.0, 20.0)
t_eval_f = np.linspace(t_span_f[0], t_span_f[1], 1001)
def F_callable_step(t): return step_force(t, F0=F0, t_step=0.0)


def F_callable_harmonic(t): return harmonic_force(
    t, F0=F0, omega=2.0, phase=0.0)


# Numerical solution for forced response
# step force
step_forced_sol_dense = solve_ivp(
    forced_ode,
    t_span_f,
    y0=[x0_f, v0_f],
    args=(m_f, c_f, k_f, F_callable_step),
    dense_output=True,
    rtol=1e-6,
    atol=1e-8,
)
# harmonic force
harmonic_force_sol_dense = solve_ivp(
    forced_ode,
    t_span_f,
    y0=[x0_f, v0_f],
    args=(m_f, c_f, k_f, F_callable_harmonic),
    dense_output=True,
    rtol=1e-6,
    atol=1e-8,
)

# Interpolate
y_f_step = step_forced_sol_dense.sol(t_eval_f)
x_f_step = y_f_step[0]
v_f_step = y_f_step[1]
y_f_harmonic = harmonic_force_sol_dense.sol(t_eval_f)
x_f_harmonic = y_f_harmonic[0]
v_f_harmonic = y_f_harmonic[1]

# print some useful calculated values
x_final_step = x_f_step[-1]
# print(f"final displacement (step force): {x_final_step}")
# print(f"F0 / k_f for comparison: {F0 / k_f}")
# peak_time_step = peak_time(t_eval_f, x_f_step)
# print(f"peak time under step force: {peak_time_step}sec")
overshoot_step = overshoot(x_f_step, x_final_step)
# print(f"overshoot under step force: {overshoot_step}")
# x_final_harmonic = x_f_harmonic[-1]
# print(f"final displacement (harmonic force): {x_final_harmonic}")
# peak_time_harmonic = peak_time(t_eval_f, x_f_harmonic)
# print(f"peak time under harmonic force: {peak_time_harmonic}sec")
# overshoot_harmonic = overshoot(x_f_harmonic, x_final_harmonic)
# print(f"overshoot under harmonic force: {overshoot_harmonic}")


# plot
# plt.plot(t_eval_f, x_f_step, 'b-', label="Step force")
# plt.plot(t_eval_f, x_f_harmonic, 'r-', label="Harmonic force")
# plt.xlabel("t")
# plt.ylabel("x(t)")
# plt.legend()
# plt.grid(True)
# plt.show()


# Exercise 6 (short)
def zero_crossing(t, y, *args) -> float:
    return y[0]


zero_crossing.terminal = False
zero_crossing.direction = 0   # both directions

x_final = F0 / k_f          # 0.25
band = 0.05 * abs(x_final)  # ±5 % of steady state


def upper_band(t, y, *args) -> float:
    return y[0] - (x_final + band)


def lower_band(t, y, *args) -> float:
    return y[0] - (x_final - band)


upper_band.terminal = False   # usually better not to stop early
lower_band.terminal = False
upper_band.direction = 0      # detect crossing from either direction
lower_band.direction = 0


step_forced_sol_events = solve_ivp(
    forced_ode,
    t_span_f,
    y0=[x0_f, v0_f],
    events=[zero_crossing, upper_band, lower_band],
    args=(m_f, c_f, k_f, F_callable_step),
    dense_output=True,
    rtol=1e-6,
    atol=1e-8,
)

# 5. Access the results
# print("t_events:", step_forced_sol_events.t_events)
# print("Number of zero crossings:", len(step_forced_sol_events.t_events[0]))
# if len(step_forced_sol_events.t_events[0]) > 0:
# print("First few zero times:", step_forced_sol_events.t_events[0][:5])

# Zero-crossing times (first event function)
# t_zeros = step_forced_sol_events.t_events[0]
# print("Zero crossings at t =", t_zeros)

# State at those zero crossings
y_zeros = step_forced_sol_events.y_events[0]   # shape (n_zeros, 2)
x_at_zeros = y_zeros[:, 0]

# Second event (enter_from_above)
t_above = step_forced_sol_events.t_events[1]
y_above = step_forced_sol_events.y_events[1]

# print(F0)
# print(m_f)
# print(k_f)

# Exercise 7 – Sweep over damping ratio ζ
zetas_arr = np.linspace(0.1, 1.5, 15)
settling_times = np.full_like(zetas_arr, np.nan, dtype=float)
peak_overshoots = np.full_like(zetas_arr, np.nan, dtype=float)

for i, zeta in enumerate(zetas_arr):
    c = 2.0 * zeta * np.sqrt(k_f * m_f)          # scalar

    # simulate the step response
    sol = solve_ivp(
        forced_ode,
        t_span_f,
        y0=[x0_f, v0_f],
        args=(m_f, c, k_f, F_callable_step),       # all scalars / callable
        t_eval=t_eval_f,                           # or dense_output=True
        rtol=1e-6,
        atol=1e-8,
    )

    x = sol.y[0]
    x_final = F0 / k_f                             # known steady-state for step
    settling_times[i] = settling_time(sol.t, x, band=0.05, x_final=x_final)
    peak_overshoots[i] = overshoot(x, x_final)     # not overshoot(x)

# settling times
# for z, ts in zip(zetas_arr, settling_times):
    # print(f"ζ = {z:.2f}  →  settling time = {ts:.3f} s")

# overshoots
# for z, ts in zip(zetas_arr, peak_overshoots):
    # print(f"ζ = {z:.2f}  →  peak overshoot = {ts:.3f}")

# plot metric versus ζ.
# plt.plot(zetas_arr, settling_times, 'b-', label="Settling times per zetas")
# plt.xlabel("ζ")
# plt.ylabel("Settling time [s]")
# plt.legend()
# plt.grid(True)
# plt.show()


# Simple optimization
# 1. Write a scalar objective function:
def settling_time_objective(zeta: float) -> float:
    """Return settling time for a given damping ratio ζ."""
    zeta = float(zeta)
    if zeta <= 0:
        return 1e6                      # large penalty for non-physical values
    c = 2.0 * zeta * np.sqrt(k_f * m_f)
    sol = solve_ivp(
        forced_ode,
        t_span_f,
        y0=[x0_f, v0_f],
        args=(m_f, c, k_f, F_callable_step),
        t_eval=t_eval_f,
        rtol=1e-6,
        atol=1e-8,
    )
    x = sol.y[0]
    ts = settling_time(sol.t, x, band=0.05, x_final=F0/k_f)
    return ts if np.isfinite(ts) else 1e6


# 2. Call a scalar minimiser, for example:
result = minimize_scalar(
    settling_time_objective,
    bounds=(0.1, 1.5),
    method="bounded",
    options={"xatol": 1e-3},
)

# print('******* Optimization Results *******')
# print("Optimal ζ:", result.x)
# print("Minimum settling time:", result.fun)
# print("Success:", result.success)
# Compare the optimal ζ with the sweep just performed — they roughly agree.
min_idx = np.argmin(settling_times)
# print(f"Optimal zeta per sweep: {zetas_arr[min_idx]}")

# EQUATIONS FOR REFERENCE;

# def underdamped_response(t, zeta, omega_n, x0, v0):
#     omega_d = omega_n * np.sqrt(1.0 - zeta**2)
#     exp_term = np.exp(-zeta * omega_n * t)
#     return exp_term * (
#         x0 * np.cos(omega_d * t)
#         + (v0 + zeta * omega_n * x0) / omega_d * np.sin(omega_d * t)
#     )

# def critical_response(t, omega_n, x0, v0):
#     return np.exp(-omega_n * t) * (x0 + (v0 + omega_n * x0) * t)

# def overdamped_response(t, zeta, omega_n, x0, v0):
#     alpha = omega_n * np.sqrt(zeta**2 - 1.0)
#     sigma = zeta * omega_n
#     A = (sigma + alpha) / (2 * alpha)
#     B = (-sigma + alpha) / (2 * alpha)
#     # (general x0, v0 version can be added later)
#     return np.exp(-sigma * t) * (A * np.exp(alpha * t) + B * np.exp(-alpha * t))
