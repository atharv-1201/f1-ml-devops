import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "f1-predictor-api"
    assert data["model_loaded"] is True

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "algorithm" in data
    assert "features" in data
    assert len(data["features"]) == 6

def test_single_prediction_valid():
    payload = {
        "driver": "VER",
        "qualifying_time": 70.669,
        "rain_probability": 0.0,
        "temperature": 22.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["driver"] == "VER"
    assert data["team"] == "Red Bull"
    assert "predicted_lap_time" in data
    assert data["predicted_lap_time"] > 0

def test_single_prediction_rain_condition():
    payload = {
        "driver": "LEC",
        "qualifying_time": 70.063,
        "rain_probability": 0.85,
        "temperature": 18.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["driver"] == "LEC"
    assert data["team"] == "Ferrari"
    assert data["predicted_lap_time"] > 0

def test_grid_simulation_default():
    payload = {
        "rain_probability": 0.1,
        "temperature": 25.0
    }
    response = client.post("/predict/grid", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["race"] == "2025 Monaco Grand Prix"
    assert len(data["podium"]) == 3
    assert len(data["standings"]) > 5
    assert data["standings"][0]["position"] == 1
    assert data["standings"][0]["gap_to_leader"] == 0.0
