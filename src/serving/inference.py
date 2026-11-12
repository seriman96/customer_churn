import os
from pathlib import Path
from functools import lru_cache

import mlflow.xgboost
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_DIR = Path(
    os.getenv(
        "MODEL_DIR",
        str(Path(__file__).resolve().parent / "model"),
    )
)

CHURN_THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.35"))

# These are the exact raw input columns used by the training dataset.
RAW_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# Binary mappings from build_features.py.
BINARY_MAPPINGS = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

# Explicit category sets are important: they ensure one-row inference
# uses the same one-hot encoding schema as training.
MULTI_CATEGORY_CATEGORIES = {
    "MultipleLines": [
        "No",
        "No phone service",
        "Yes",
    ],
    "InternetService": [
        "DSL",
        "Fiber optic",
        "No",
    ],
    "OnlineSecurity": [
        "No",
        "No internet service",
        "Yes",
    ],
    "OnlineBackup": [
        "No",
        "No internet service",
        "Yes",
    ],
    "DeviceProtection": [
        "No",
        "No internet service",
        "Yes",
    ],
    "TechSupport": [
        "No",
        "No internet service",
        "Yes",
    ],
    "StreamingTV": [
        "No",
        "No internet service",
        "Yes",
    ],
    "StreamingMovies": [
        "No",
        "No internet service",
        "Yes",
    ],
    "Contract": [
        "Month-to-month",
        "One year",
        "Two year",
    ],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


# ---------------------------------------------------------------------
# Model and feature metadata loading
# ---------------------------------------------------------------------

def load_feature_columns() -> list[str]:
    columns_path = MODEL_DIR / "feature_columns.txt"

    if not columns_path.is_file():
        raise FileNotFoundError(
            f"Feature metadata not found: {columns_path}. "
            "Run scripts/package_model.py first."
        )

    columns = [
        line.strip()
        for line in columns_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(columns) != 30:
        raise ValueError(
            f"Expected 30 model features, but found {len(columns)}."
        )

    return columns


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}. "
            "Run scripts/package_model.py first."
        )

    if not (MODEL_DIR / "MLmodel").is_file():
        raise FileNotFoundError(
            f"MLflow MLmodel file not found in {MODEL_DIR}."
        )

    model = mlflow.xgboost.load_model(str(MODEL_DIR))

    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba().")

    return model


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def _validate_input(customer: dict) -> pd.DataFrame:
    missing = [column for column in RAW_FEATURES if column not in customer]

    if missing:
        raise ValueError(f"Missing customer fields: {missing}")

    df = pd.DataFrame(
        [{column: customer[column] for column in RAW_FEATURES}]
    )

    # Normalize string values.
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].astype("string").str.strip()

    # Validate and convert numeric fields.
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

        if df[column].isna().any():
            raise ValueError(
                f"'{column}' must contain a valid numeric value."
            )

    # Match preprocess.py.
    df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    if df["SeniorCitizen"].iloc[0] not in (0, 1):
        raise ValueError("SeniorCitizen must be either 0 or 1.")

    for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if df[column].iloc[0] < 0:
            raise ValueError(f"'{column}' cannot be negative.")

    # Validate binary categories before applying training mappings.
    for column, mapping in BINARY_MAPPINGS.items():
        value = df[column].iloc[0]

        if pd.isna(value) or value not in mapping:
            raise ValueError(
                f"Invalid value '{value}' for '{column}'. "
                f"Expected one of {list(mapping)}."
            )

    # Apply the exact binary mappings from build_features.py.
    for column, mapping in BINARY_MAPPINGS.items():
        df[column] = df[column].map(mapping).astype(int)

    # Validate multi-category values and assign the complete category
    # list before one-hot encoding. This prevents single-row encoding
    # from dropping categories that are absent from that customer.
    for column, categories in MULTI_CATEGORY_CATEGORIES.items():
        value = df[column].iloc[0]

        if pd.isna(value) or value not in categories:
            raise ValueError(
                f"Invalid value '{value}' for '{column}'. "
                f"Expected one of {categories}."
            )

        df[column] = pd.Categorical(
            df[column],
            categories=categories,
        )

    return df


def build_customer_features(customer: dict) -> pd.DataFrame:
    df = _validate_input(customer)

    # Match training: drop_first=True, using the complete known
    # categories rather than only the values in this single row.
    df = pd.get_dummies(
        df,
        columns=list(MULTI_CATEGORY_CATEGORIES.keys()),
        drop_first=True,
        dtype=int,
    )

    feature_columns = load_feature_columns()

    # Restore exact training feature order and fill absent dummy
    # columns with zero.
    df = df.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    # Ensure XGBoost receives numeric values.
    #df = df.astype("int64")
    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df.shape[1] != 30:
        raise ValueError(
            f"Expected 30 features, received {df.shape[1]}."
        )

    return df


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def predict(customer: dict) -> dict:
    """
    Predict customer churn.

    Returns:
        prediction: Human-readable prediction.
        churn_probability: Probability of churn.
        threshold: Decision threshold used.
    """

    if not 0 < CHURN_THRESHOLD < 1:
        raise ValueError("CHURN_THRESHOLD must be between 0 and 1.")

    features = build_customer_features(customer)
    model = load_model()

    probabilities = model.predict_proba(features)

    # Locate the probability corresponding to class 1 (Churn).
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("The trained model does not contain churn class 1.")

    churn_index = classes.index(1)
    churn_probability = float(probabilities[0][churn_index])

    is_churn = churn_probability >= CHURN_THRESHOLD

    return {
        "prediction": (
            "Likely to churn" if is_churn else "Not likely to churn"
        ),
        "churn_probability": round(churn_probability, 4),
        "threshold": CHURN_THRESHOLD,
    }