from .forces import spring_force, damper_force, natural_frequency, damping_ratio


class MassSpringSystem:
    def __init__(self, m: float, k: float, c: float = 0.0) -> None:
        """Create a mass-spring-damper system.

        Parameters
        ----------
        m : float
            Mass (must be > 0)
        k : float
            Spring stiffness (must be > 0)
        c : float, optional
            Damping coefficient (must be ≥ 0). Default is 0 (undamped).
        """
        if m <= 0 or k <= 0:
            raise ValueError("mass and stiffness must be positive values")
        if c < 0:
            raise ValueError("damping must be a non-negative value")

        self._m = m
        self._k = k
        self._c = c

    def __repr__(self) -> str:
        """Return official representation of MassSpringSystem class"""
        return f"MassSpringSystem(m={self.m}, k={self.k}, c={self.c})"

    @property
    def m(self) -> float:
        """Return current mass"""
        return self._m

    @m.setter
    def m(self, new_m: float) -> None:
        """Validate and set new mass value"""
        if new_m <= 0:
            raise ValueError("mass must be a positive value")
        self._m = float(new_m)

    @property
    def k(self) -> float:
        """Return current stiffness"""
        return self._k

    @k.setter
    def k(self, new_k: float) -> None:
        """Validate and set new stiffness value"""
        if new_k <= 0:
            raise ValueError("stiffness must be a positive value")
        self._k = float(new_k)

    @property
    def c(self) -> float:
        """Return current damping value"""
        return self._c

    @c.setter
    def c(self, new_c: float) -> None:
        """Validate and set new damping value"""
        if new_c < 0:
            raise ValueError("damping must be a non-negative value")
        self._c = float(new_c)

    def force(self, x: float, v: float) -> float:
        """Return total force F = –k x – c v, consisting of restoring and damping forces"""
        return spring_force(self.k, x) + damper_force(self.c, v)

    @property
    def omega_n(self) -> float:
        """Undamped natural frequency (rad/s)."""
        return natural_frequency(self.m, self.k)

    @property
    def zeta(self) -> float:
        """Damping ratio ζ."""
        return damping_ratio(self.m, self.c, self.k)
