"""
Model Training Script for CI/CD Pipeline.
Executes data preprocessing, trains GradientBoostingRegressor,
evaluates Mean Absolute Error, and exports serialized artifact.
"""
import os
import sys
from app.model import F1PredictionEngine

def run_training():
    print("🏎️ Starting F1 Prediction Model Training Pipeline...")
    model_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    engine = F1PredictionEngine(model_path=model_path)
    engine.train_baseline()

    print(f"✅ Model successfully trained and saved to: {model_path}")
    print(f"📊 Metadata: {engine.metadata}")

    # Sanity inference check
    sample_pred = engine.predict_driver("VER", qualifying_time=70.669)
    print(f"🧪 Sanity Check - VER Predicted Lap: {sample_pred['predicted_lap_time']}s")
    assert sample_pred["predicted_lap_time"] > 0, "Invalid predicted lap time"
    print("🚀 Training and validation complete!")

if __name__ == "__main__":
    run_training()
