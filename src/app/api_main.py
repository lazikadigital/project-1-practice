from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.physics_utils import natural_frequency, damping_ratio
from app.demo_scipy_ode import forced_ode, free_response_ode, peak_time, overshoot, settling_time
from scipy.integrate import solve_ivp
import numpy as np
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware


# EXERCISE 1. INITIAL SETUP AND TESTING
app = FastAPI(title="Mass-Spring Simulator API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000",
                   "https://our-production-frontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SystemParams(BaseModel):
    m: float = Field(gt=0, description="Mass")
    k: float = Field(gt=0, description="Stiffness")
    c: float = Field(ge=0, description="Damping")


class SystemProperties(BaseModel):
    omega_n: float
    zeta: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/system/properties", response_model=SystemProperties)
def system_properties(params: SystemParams) -> SystemProperties:
    return SystemProperties(
        omega_n=natural_frequency(params.m, params.k),
        zeta=damping_ratio(params.m, params.c, params.k)
    )


# Exercise 2 – Simulation endpoint
class SimulateRequest(BaseModel):
    m: float = Field(gt=0)
    k: float = Field(gt=0)
    c: float = Field(ge=0)
    x0: float = 0.0
    v0: float = 0.0
    t_start: float = 0.0
    t_end: float = Field(gt=0, default=15.0)
    n_points: int = Field(gt=2, default=501)


class SimulateResponse(BaseModel):
    t: list[float]
    x: list[float]
    v: list[float]
    omega_n: float
    zeta: float
    peak_time: float
    overshoot: float
    settling_time: float | None


@app.post("/simulate", response_model=SimulateResponse)
def simulate(params: SimulateRequest) -> SimulateResponse:
    if params.t_end <= params.t_start:
        raise HTTPException(
            status_code=422, detail="t_end must be greater than t_start")

    t_eval = np.linspace(params.t_start, params.t_end, params.n_points)

    sol = solve_ivp(
        free_response_ode,
        t_span=(params.t_start, params.t_end),
        y0=[params.x0, params.v0],
        t_eval=t_eval,
        args=(params.m, params.c, params.k),
        rtol=1e-6,
        atol=1e-8,
    )

    if not sol.success:
        raise HTTPException(
            status_code=400,
            detail=f"Solver failed: {sol.message}",
        )

    t_arr = sol.t
    x_arr = sol.y[0]
    v_arr = sol.y[1]

    omega_n = natural_frequency(params.m, params.k)
    zeta = damping_ratio(params.m, params.c, params.k)

    t_peak = peak_time(t_arr, x_arr)
    abs_overshoot = overshoot(x_arr, x_final=0.0)
    t_settle = settling_time(t_arr, x_arr, band=0.05, x_final=0.0)

    return SimulateResponse(
        t=t_arr.tolist(),
        x=x_arr.tolist(),
        v=v_arr.tolist(),
        omega_n=omega_n,
        zeta=zeta,
        peak_time=float(t_peak),
        overshoot=float(abs_overshoot),
        settling_time=None if t_settle is None or not np.isfinite(
            t_settle) else float(t_settle),
    )


# Exercise 3 – Error handling + CORS

# sol = solve_ivp(
#     free_response_ode,
#     t_span,
#     y0=[x0, v0],
#     t_eval=t_eval,
#     args=(m, c, k),
#     rtol=1e-6,
#     atol=1e-8,
# )

# step_forced_sol_dense = solve_ivp(
#     forced_ode,
#     t_span_f,
#     y0=[x0_f, v0_f],
#     args=(m_f, c_f, k_f, F_callable_step),
#     dense_output=True,
#     rtol=1e-6,
#     atol=1e-8,
# )

# IMPORTANT!!!
# How you launch the server matters for imports. Typical patterns:
# from the project root, with src layout
# uvicorn app.main:app --reload --app-dir src
# or
# cd src && uvicorn app.main:app --reload
