import numpy as np

# Exercise A – Multiple natural frequencies
t = np.linspace(0, 10, 1001)
omegas = np.array([1, 2, 3, 4, 5])
omegas_col = omegas[:, np.newaxis]
# print(f"omegas col: {omegas_col.shape}")
x = np.cos(omegas_col * t)
# print(f"x(t): {x.shape}")

# Print the shape and the value at t ≈ 1.0 s for every ωₙ.
time_mask = np.isclose(t, 1.0)
x_t1 = x[:, time_mask]
# print(f"shape: {x_t1.shape}")
print(f"at omega = 1: {x_t1[0][0]}")
print(f"at omega = 2: {x_t1[1][0]}")
print(f"at omega = 3: {x_t1[2][0]}")
print(f"at omega = 4: {x_t1[3][0]}")
print(f"at omega = 5: {x_t1[4][0]}")

# Exercise B – Multiple initial conditions
omega_n = 2.0
x0 = np.array([0.0, 0.25, 1.0, 2.0, 5.0])
v0 = np.array([0.0, 1.0, 2.5, 5.0, 7.5])
x0_col = x0[:, np.newaxis]
v0_col = v0[:, np.newaxis]
x_full = x0_col * np.cos(omega_n * t) + v0_col * np.sin(omega_n * t) / omega_n

# Exercise C – Simple parameter sweep over damping ratio
zeta = np.linspace(0.0, 1.5, 16)

omega_d = np.zeros_like(zeta)

under = zeta < 1.0
critical = np.isclose(zeta, 1.0)
over = zeta > 1.0

# Under-damped
omega_d[under] = omega_n * np.sqrt(1.0 - zeta[under]**2)
# Critically damped
omega_d[critical] = 0.0
# Over-damped – store the real coefficient
omega_d[over] = omega_n * np.sqrt(zeta[over]**2 - 1.0)

# Exercise D
zetas = np.array([0.2, 1.0, 1.5])
x_0 = 1.0
v_0 = 0.0
t_arr = np.linspace(0, 15, 1501)

under_damped = zetas < 1
critically_damped = np.isclose(zetas, 1.0)
over_damped = zetas > 1

omega_damped = omega_n * np.sqrt(1.0 - np.square(zetas[under_damped]))

trajectories = np.zeros((len(zetas), len(t_arr)))


# the helper-function style (Pattern 1) remains the most future-proof when the number of trajectories per regime can grow.
def underdamped_response(t, zeta, omega_n, x0, v0):
    omega_d = omega_n * np.sqrt(1.0 - zeta**2)
    exp_term = np.exp(-zeta * omega_n * t)
    return exp_term * (
        x0 * np.cos(omega_d * t)
        + (v0 + zeta * omega_n * x0) / omega_d * np.sin(omega_d * t)
    )


def critical_response(t, omega_n, x0, v0):
    return np.exp(-omega_n * t) * (x0 + (v0 + omega_n * x0) * t)


def overdamped_response(t, zeta, omega_n, x0, v0):
    alpha = omega_n * np.sqrt(zeta**2 - 1.0)
    sigma = zeta * omega_n
    A = (sigma + alpha) / (2 * alpha)
    B = (-sigma + alpha) / (2 * alpha)
    # (general x0, v0 version can be added later)
    return np.exp(-sigma * t) * (A * np.exp(alpha * t) + B * np.exp(-alpha * t))


# Populate each row (trajectory) by calling the helper functions through iteration
for i, z in enumerate(zetas):
    if z < 1.0:
        trajectories[i] = underdamped_response(t_arr, z, omega_n, x_0, v_0)
    elif np.isclose(z, 1.0):
        trajectories[i] = critical_response(t_arr, omega_n, x_0, v_0)
    else:
        trajectories[i] = overdamped_response(t_arr, z, omega_n, x_0, v_0)


# Compute the threshold per trajectory (so the biggest amplitude one does not distort others):
peak_per_trajectory = np.max(
    np.abs(trajectories), axis=1)          # shape (3,)
threshold = 0.05 * peak_per_trajectory[:,
                                       np.newaxis]               # shape (3, 1)

# Build a boolean mask:
mask = np.abs(trajectories) <= threshold
# Find the first index where the mask is True.

# mask shape (3, N)
has_entered = np.any(mask, axis=1)
# still works when has_entered is True
first_idx = np.argmax(mask, axis=1)
first_idx = np.where(has_entered, first_idx, -1)

first_entry_times = np.full(len(zetas), np.nan)
first_entry_times[has_entered] = t_arr[first_idx[has_entered]]

# print(first_entry_times)

print("First time entering ±5 % band:")
for index, zeta in enumerate(zetas):
    print(f"ζ = {zeta} : {first_entry_times[index]}s")
