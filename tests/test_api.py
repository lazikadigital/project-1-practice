from fastapi.testclient import TestClient
from app.api_main import app   # adjust import to your module path

print("line 1 reads")

client = TestClient(app)

print("line 2 reads")


def test_simulate_underdamped():
    r = client.post("/simulate", json={
        "m": 1.0, "k": 4.0, "c": 0.5,
        "x0": 1.0, "v0": 0.0,
        "t_start": 0.0, "t_end": 15.0, "n_points": 101,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["omega_n"] == 2.0
    assert abs(data["zeta"] - 0.125) < 1e-9
    assert len(data["t"]) == 101
    assert abs(data["x"][0] - 1.0) < 1e-6
