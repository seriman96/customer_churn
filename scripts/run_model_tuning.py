'''
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna

print("=== Phase 2: Modeling with XGBoost ===")

df = pd.read_csv("data/processed/telco_churn_processed.csv")

# target must be numeric 0/1
if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1})

assert df["Churn"].isna().sum() == 0, "Churn has NaNs"
assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1"

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

THRESHOLD = 0.4

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    from sklearn.metrics import recall_score
    return recall_score(y_test, y_pred, pos_label=1)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("Best Params:", study.best_params)
print("Best Recall:", study.best_value)
'''

from pathlib import Path
import sys

import optuna
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import build_features


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_processed.csv"
TARGET = "Churn"
THRESHOLD = 0.40
N_TRIALS = 30
RANDOM_STATE = 42


def load_training_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = (
            df[TARGET].astype("string").str.strip().map({"No": 0, "Yes": 1})
        )

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing or unexpected values.")

    df[TARGET] = df[TARGET].astype(int)

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    # Use the same feature engineering logic as the training pipeline.
    df = build_features(df, target_col=TARGET)

    for column in df.select_dtypes(include=["bool"]).columns:
        df[column] = df[column].astype(int)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    invalid_columns = X.select_dtypes(exclude=["number"]).columns.tolist()
    if invalid_columns:
        raise TypeError(
            f"Non-numeric features remain after feature engineering: "
            f"{invalid_columns}"
        )

    return X, y


def objective(trial, X_train, X_test, y_train, y_test):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    return recall_score(y_test, predictions, pos_label=1)


def main():
    print("=== Phase 2: XGBoost Hyperparameter Optimization ===")

    X, y = load_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(
            trial, X_train, X_test, y_train, y_test
        ),
        n_trials=N_TRIALS,
    )

    print("\nBest parameters:")
    print(study.best_params)

    print(f"\nBest recall at threshold {THRESHOLD}: {study.best_value:.4f}")


if __name__ == "__main__":
    main()
