import numpy as np


# 1.1 Dot product and matrix multiplication

# Create two vectors and a small matrix that could represent
# a 2×2 state-space system (or a simple transformation)
v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([4.0, 5.0, 6.0])

A = np.array([[0.0, 1.0],
              [-4.0, -0.5]])   # example [0 1; -ωₙ² -2ζωₙ]

dot_product_1 = np.dot(v1, v2)
print(f"dot_product_1: {dot_product_1}")
dot_product_2 = v1 @ v2
print(f"dot_product_2: {dot_product_2}")

# generating some_2d_vector (matrix) and filling it with random integers 1 to 11
rng = np.random.default_rng()
some_2d_vector = rng.integers(low=1, high=11, size=(2, 2))
print(some_2d_vector)

random_product = A @ some_2d_vector
print(random_product)
matrix_square = A @ A
print(matrix_square)


# 1.2 Vector and matrix norms
# A simple 1-D array of displacement trajectory
x = np.linspace(0, 100, 50)
l2_norm = np.linalg.norm(x)
print(l2_norm)
inf_norm = np.linalg.norm(x, ord=np.inf)
print(inf_norm)
frob_norm = np.linalg.norm(matrix_square)
print(frob_norm)


# 1.3 Solving a linear system
# Solve A x = b for a 2×2 system
A_2 = np.array([[4.0, 1.0],
                [1.0, 3.0]])
b = np.array([1.0, 2.0])
# - Solve with np.linalg.solve
x_2 = np.linalg.solve(A_2, b)
print(x_2)
# - Verify by computing A @ x and comparing to b
verified_b = A_2 @ x_2
print(verified_b)
# - (Optional) Compute the inverse and multiply – compare numerical quality
x_2_alt = np.linalg.inv(A_2) @ b
print(x_2_alt)


# 1.4 Mini physics application
# Kinetic + potential energy of the mass-spring system at one instant
m = 1.0
k = 4.0
x = 0.5
v = -0.2

E_arithm = 0.5 * m * v**2 + 0.5 * k * x**2
print(E_arithm)


# x and v are now 1-D arrays of the same length
# or 0.5 * m * np.dot(v, v) for a single scalar energy
E_kinetic = 0.5 * m * v * v
E_potential = 0.5 * k * x * x
E_total = E_kinetic + E_potential  # still an array
# or, if you ever store the state as a vector [x, v]:
state = np.array([x, v])
# … more advanced constructions with quadratic forms, etc.


# 2.1 Single array
t = np.linspace(0, 10, 1001)
x_3 = 0.5 * np.cos(2.0 * t)
np.save("displacement.npy", x_3)
loaded_x_3 = np.load("displacement.npy")
print(np.allclose(x_3, loaded_x_3, rtol=1e-05, atol=1e-08, equal_nan=False))


# 2.2 Multiple arrays in one file
# Save t, x and a force array together
f = np.linspace(0, 50, 25)
np.savez("simulation_results.npz", time=t, displacement=x_3, force=f)

# Load and access by name
data = np.load("simulation_results.npz")
t2 = data["time"]


# 2.3 Human-readable text (optional but useful)
# Save a small table as CSV-style text
np.savetxt("results.csv", np.column_stack((t[::10], x_3[::10])),
           header="time, displacement", delimiter=",", comments="")

# Load it back with np.loadtxt
loaded_results = np.loadtxt("results.csv", delimiter=',', skiprows=1)
print(loaded_results.shape)


# 2.4 Round-trip test
time_data = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
states_data = np.array([False, False, True, True, True, True, False])
metrics_data = np.array([0.0, 2.25, 3.37, 5.15, 4.01, 3.07, 1.95])
input_data = {
    "time": time_data,
    "states": states_data,
    "metrics": metrics_data
}

np.savez("input_data.npz", time=time_data,
         states=states_data, metrics=metrics_data)
output_data = np.load("input_data.npz")
# assertions
print(np.allclose(
    time_data, output_data["time"], rtol=1e-05, atol=1e-08, equal_nan=False))
print(np.allclose(states_data,
      output_data["states"], rtol=1e-05, atol=1e-08, equal_nan=False))
print(np.allclose(metrics_data,
      output_data["metrics"], rtol=1e-05, atol=1e-08, equal_nan=False))
