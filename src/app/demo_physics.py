from .physics_utils import MassSpringSystem, spring_force, damper_force


# 1. Create systems
undamped = MassSpringSystem(m=1.0, k=4.0)          # c = 0
under_damped = MassSpringSystem(m=1.0, k=4.0, c=1.0)  # c < 2√(k * m)
critically_damped = MassSpringSystem(m=1.0, k=1.0, c=2.0)  # c = 2√(k * m)
over_damped = MassSpringSystem(m=1.0, k=1.0, c=3.0)  # c > 2√(k * m)


# 2. Print info
print("Undamped system")
print("-----------------")
print(undamped)
print(f"ωₙ = {undamped.omega_n:.4f} rad/s, ζ = {undamped.zeta:.4f}\n")
print("Under-damped system")
print("-----------------")
print(under_damped)
print(f"ωₙ = {under_damped.omega_n:.4f} rad/s, ζ = {under_damped.zeta:.4f}\n")
print("Critically damped system")
print("-----------------")
print(critically_damped)
print(
    f"ωₙ = {critically_damped.omega_n:.4f} rad/s, ζ = {critically_damped.zeta:.4f}\n")
print("Over-damped system")
print("-----------------")
print(over_damped)
print(f"ωₙ = {over_damped.omega_n:.4f} rad/s, ζ = {over_damped.zeta:.4f}\n")

# 3. Compare pure functions vs class
x, v = 0.5, 0.1
print("Force via class :", under_damped.force(x, v))
print("Force via funcs :", spring_force(
    under_damped.k, x) + damper_force(under_damped.c, v))

# 4. Test validation
try:
    bad = MassSpringSystem(m=-1.0, k=10.0)
except ValueError as e:
    print("Caught expected error:", e)
