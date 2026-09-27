"""
modeling.py
===========
Trains and evaluates the shipping-lead-time predictive models described in
the project spec: Linear Regression (baseline), Random Forest Regressor,
Gradient Boosting Regressor. Selects the best model on held-out RMSE/MAE/R2
and persists it (plus the fitted encoders) for reuse by simulation.py and
the Streamlit app.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
NUMERIC_FEATURES = ["DistanceMiles", "Units", "UnitCost"]
CATEGORICAL_FEATURES = ["Product Name", "Factory", "Region", "Ship Mode", "Division"]
TARGET = "LeadTimeDays"


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def _candidate_models() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = {}
    fitted = {}
    for name, model in _candidate_models().items():
        pipe = Pipeline([("prep", _build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        results[name] = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}
        fitted[name] = pipe

    # best-performing model selected on lowest RMSE, tie-broken by higher R2
    best_name = min(results, key=lambda k: (results[k]["RMSE"], -results[k]["R2"]))
    best_pipe = fitted[best_name]

    return best_name, best_pipe, results, (X_test, y_test)


def save_model(pipe, path: Path = ARTIFACTS_DIR / "lead_time_model.pkl"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipe, f)


def load_model(path: Path = ARTIFACTS_DIR / "lead_time_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_lead_time(pipe, rows: pd.DataFrame) -> np.ndarray:
    """rows must contain NUMERIC_FEATURES + CATEGORICAL_FEATURES columns."""
    return pipe.predict(rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES])


if __name__ == "__main__":
    from data_pipeline import build_dataset

    df = build_dataset()
    best_name, best_pipe, results, _ = train_and_evaluate(df)
    print("Model comparison:")
    for name, m in results.items():
        flag = "  <-- selected" if name == best_name else ""
        print(f"  {name:>18}: RMSE={m['RMSE']:.3f} days | MAE={m['MAE']:.3f} days | R2={m['R2']:.3f}{flag}")

    save_model(best_pipe)
    with open(ARTIFACTS_DIR / "model_metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    print(f"\nBest model: {best_name} -> saved to artifacts/lead_time_model.pkl")
