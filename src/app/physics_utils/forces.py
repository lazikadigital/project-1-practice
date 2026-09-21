import math


def spring_force(k: float, x: float) -> float:
    """Return the spring restoring force.

    Parameters
    ----------
    k : float
        Spring stiffness
    x : float
        Displacement from equilibrium

    Returns
    -------
    float
        F = –k * x
    """
    return -k * x


def damper_force(c: float, v: float) -> float:
    """Return the damping force.

    Parameters
    ---------- 
    c : float 
       Damping coefficient
    v : float
       Velocity   

    Returns
    ------- 
    float 
        F = -c * v    
    """
    return -c * v


def natural_frequency(m: float, k: float) -> float:
    """Return the undamped natural frequency ωₙ in rad/s.

    Parameters
    ----------
    m : float
        Mass (> 0)
    k : float
        Spring stiffness (> 0)

    Returns
    -------
    float
        ωₙ = √(k / m)
    """
    return math.sqrt(k / m)


def damping_ratio(m: float, c: float, k: float) -> float:
    """Return the damping ratio ζ (dimensionless).

    ζ < 1 → underdamped, ζ = 1 → critically damped, ζ > 1 → overdamped.
    """
    if m <= 0 or k <= 0:
        raise ValueError("m and k must be positive")
    return c / (2 * math.sqrt(k * m))
