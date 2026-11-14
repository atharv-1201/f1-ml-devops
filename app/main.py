import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    GridPredictionRequest,
    GridPredictionResponse,
    HealthResponse
)
from app.model import F1PredictionEngine

app = FastAPI(
    title="🏎️ FormulaVision API",
    description="End-to-End MLOps & DevOps API for Formula 1 race pace prediction and podium simulation.",
    version="1.1.0"
)

# Enable CORS for external frontend or Swagger access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML Prediction Engine
engine = F1PredictionEngine()

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", tags=["UI"])
def serve_dashboard():
    """Serves the interactive web UI dashboard."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "F1 Prediction API is running. Visit /docs for Swagger API documentation."}

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """
    Health check endpoint for Docker HEALTHCHECK and AWS load balancer / EC2 monitoring.
    """
    is_ready = engine.model is not None
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        service="f1-predictor-api",
        version="1.0.0",
        model_loaded=is_ready
    )

@app.get("/model-info", tags=["Monitoring"])
def get_model_info():
    """Returns trained model metadata and feature specifications."""
    return {
        "service": "f1-predictor-api",
        "algorithm": "GradientBoostingRegressor",
        "features": [
            "QualifyingTime",
            "RainProbability",
            "Temperature",
            "TeamPerformanceScore",
            "CleanAirRacePace (s)",
            "AveragePositionChange"
        ],
        "metadata": engine.metadata
    }

@app.post("/predict", response_model=PredictionResponse, tags=["ML Inference"])
def predict_single(payload: PredictionRequest):
    """
    Inference endpoint: Predicts expected lap time for a single driver given qualifying & weather conditions.
    """
    try:
        res = engine.predict_driver(
            driver=payload.driver,
            qualifying_time=payload.qualifying_time,
            rain_probability=payload.rain_probability,
            temperature=payload.temperature
        )
        return PredictionResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/predict/grid", response_model=GridPredictionResponse, tags=["ML Inference"])
def predict_grid(payload: GridPredictionRequest):
    """
    Simulates the entire 2025 Monaco GP grid, returning full standings and podium finishers.
    """
    try:
        res = engine.predict_grid(
            rain_probability=payload.rain_probability,
            temperature=payload.temperature,
            custom_qualifying=payload.custom_qualifying
        )
        return GridPredictionResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grid simulation error: {str(e)}")
