"""Placing customers into segments with the saved K-Means model."""
import json

import joblib

from .config import MODEL_DIR
from .data_prep import prepare_model_input


def load_artifacts():
    """Load everything the notebook saved: the preparation recipe, the scaler, the model and the segment names."""
    preprocessing_config = json.loads((MODEL_DIR / "preprocessing_config.json").read_text())
    model_config = json.loads((MODEL_DIR / "model_config.json").read_text())
    return {
        "preprocessing_config": preprocessing_config,
        "scaler": joblib.load(MODEL_DIR / "scaler.joblib"),
        "model": joblib.load(MODEL_DIR / "kmeans_final.joblib"),
        "segment_names": {int(k): v for k, v in model_config["segment_names"].items()},
    }


def assign_segments(customers, artifacts=None):
    """Return a copy of the customer table with 'Cluster' and 'Segment' columns added.

    `customers` needs one row per customer and the selected feature columns in everyday units
    (days, orders, pounds), as produced by `build_customer_features`.
    """
    artifacts = artifacts or load_artifacts()
    model_input = prepare_model_input(customers, artifacts["preprocessing_config"], artifacts["scaler"])
    result = customers.copy()
    result["Cluster"] = artifacts["model"].predict(model_input)
    result["Segment"] = result["Cluster"].map(artifacts["segment_names"])
    return result
