import os

from joblib import dump, load

from churn_platform.data.loader import get_project_root


def get_model_path(filename: str = "logreg_baseline.joblib") -> str:
    """
    Build the full path to the model file, under project_root/models/.
    """
    root_dir = get_project_root()
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, filename)

    return model_path


def save_model(model, filename: str = "logreg_baseline.joblib") -> str:
    """
    Save a model to disk using joblib.
    """
    path = get_model_path(filename)
    dump(model, path)
    print(f"Model saved to: {path}")
    return path


def load_model(filename: str = "logreg_baseline.joblib"):
    """
    Load a model from disk.
    """
    path = get_model_path(filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    model = load(path)
    print(f"Model loaded from: {path}")
    return model
