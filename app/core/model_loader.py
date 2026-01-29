# app/core/model_loader.py
import joblib
from pathlib import Path

def load_artifacts(models_dir: Path):
    best_model_name = (models_dir / "best_model_name.txt").read_text().strip()
    model = joblib.load(models_dir / f"{best_model_name.lower()}_pipeline.pkl")
    label_encoder = joblib.load(models_dir / "label_encoder.pkl")
    feature_names = joblib.load(models_dir / "feature_names.pkl")

    return model, label_encoder, feature_names
