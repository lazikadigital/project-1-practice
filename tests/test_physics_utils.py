import pytest
from app.physics_utils import (
    spring_force,
    damper_force,
    natural_frequency,
    damping_ratio,
    MassSpringSystem,
)


# ON GIT COMMIT, SAY I ADDED PYPROJECT.TOML, INSTALLED PACKAGES (DEPENDENCIES),
# ADDED TEST DIR STRUCTURE AND TEST FUNCTIONS
# NEXT BRANCH WILL BE CALLED NUMPY

# COMMAND TO RUN THE TESTS: pytest tests/ -v
# RUN pip install -e TO INSTALL EVERYTHING IN pyproject.toml


def test_spring_force():
    assert spring_force(k=10.0, x=0.5) == pytest.approx(-5.0)


def test_damper_force():
    assert damper_force(c=2.0, v=0.3) == pytest.approx(-0.6)


def test_natural_frequency():
    # m=1, k=4 → ωₙ = 2
    assert natural_frequency(m=1.0, k=4.0) == pytest.approx(2.0)


def test_damping_ratio():
    # m=1, k=1, c=2 → ζ = 1.0 (critical)
    assert damping_ratio(m=1.0, c=2.0, k=1.0) == pytest.approx(1.0)


def test_mass_spring_system_properties():
    sys = MassSpringSystem(m=1.0, k=4.0, c=1.0)
    assert sys.m == 1.0
    assert sys.k == 4.0
    assert sys.c == 1.0
    assert sys.omega_n == pytest.approx(2.0)
    assert sys.zeta == pytest.approx(0.25)


def test_force_method_matches_pure_functions():
    sys = MassSpringSystem(m=1.0, k=4.0, c=0.5)
    x, v = 0.25, 0.1
    expected = spring_force(sys.k, x) + damper_force(sys.c, v)
    assert sys.force(x, v) == pytest.approx(expected)


def test_invalid_mass_raises():
    with pytest.raises(ValueError, match="positive"):
        MassSpringSystem(m=-1.0, k=10.0)


def test_invalid_damping_raises():
    with pytest.raises(ValueError, match="non-negative"):
        MassSpringSystem(m=1.0, k=10.0, c=-0.1)
