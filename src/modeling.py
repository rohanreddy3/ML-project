from pathlib import Path
import joblib

def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved to: {path}")

def load_model(path):
    return joblib.load(path)
