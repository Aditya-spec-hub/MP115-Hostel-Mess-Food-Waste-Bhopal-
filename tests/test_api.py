import pytest
from fastapi.testclient import TestClient

import app.main as main

GOOD = {"students_registered": 500, "attendance_pct": 70.0, "menu_rating": 4.0,
        "temp_c": 25.0, "cooked_kg": 120.0, "plate_size_cm": 27.0,
        "meal": "lunch", "hostel": "Boys Hostel 1"}


class DummyModel:
    def predict(self, X):
        return [42.0] * len(X)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "model", DummyModel())
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert r.json()["prediction"] == 42.0


def test_cooked_kg_above_1000_rejected(client):
    r = client.post("/predict", json={**GOOD, "cooked_kg": 1001})
    assert r.status_code == 422


def test_plate_size_below_10_rejected(client):
    r = client.post("/predict", json={**GOOD, "plate_size_cm": 9.5})
    assert r.status_code == 422


def test_missing_field_rejected(client):
    bad = {k: v for k, v in GOOD.items() if k != "meal"}
    assert client.post("/predict", json=bad).status_code == 422
