
import pandas as pd
import pytest

from src.serving.inference import (
    RAW_FEATURES,
    CHURN_THRESHOLD,
    load_feature_columns,
    load_model,
    build_customer_features,
    predict,
)


def test_raw_feature_schema_contains_19_features():
    assert len(RAW_FEATURES) == 19
    assert len(set(RAW_FEATURES)) == 19


def test_model_feature_metadata_contains_30_features():
    columns = load_feature_columns()

    assert len(columns) == 30
    assert len(set(columns)) == 30


def test_model_loads_successfully():
    model = load_model()

    assert model is not None
    assert hasattr(model, "predict_proba")
    assert 1 in list(model.classes_)


def test_customer_features_have_expected_shape(valid_customer):
    features = build_customer_features(valid_customer)

    assert features.shape == (1, 30)


def test_customer_features_match_model_schema(valid_customer):
    features = build_customer_features(valid_customer)
    expected_columns = load_feature_columns()

    assert features.columns.tolist() == expected_columns


def test_customer_features_are_numeric(valid_customer):
    features = build_customer_features(valid_customer)

    assert all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in features.dtypes
    )


def test_binary_features_are_encoded_correctly(valid_customer):
    features = build_customer_features(valid_customer)

    assert features.loc[0, "gender"] == 1
    assert features.loc[0, "Partner"] == 1
    assert features.loc[0, "Dependents"] == 0
    assert features.loc[0, "PhoneService"] == 1
    assert features.loc[0, "PaperlessBilling"] == 1


def test_numeric_features_are_preserved(valid_customer):
    features = build_customer_features(valid_customer)

    assert features.loc[0, "SeniorCitizen"] == 0
    assert features.loc[0, "tenure"] == 12
    assert features.loc[0, "MonthlyCharges"] == 70.0
    assert features.loc[0, "TotalCharges"] == 840.0


def test_missing_customer_field_is_rejected(valid_customer):
    del valid_customer["SeniorCitizen"]

    with pytest.raises(ValueError, match="Missing customer fields"):
        build_customer_features(valid_customer)


def test_invalid_binary_category_is_rejected(valid_customer):
    valid_customer["gender"] = "Unknown"

    with pytest.raises(ValueError, match="Invalid value"):
        build_customer_features(valid_customer)


def test_invalid_multi_category_value_is_rejected(valid_customer):
    valid_customer["Contract"] = "Three years"

    with pytest.raises(ValueError, match="Invalid value"):
        build_customer_features(valid_customer)


def test_negative_tenure_is_rejected(valid_customer):
    valid_customer["tenure"] = -5

    with pytest.raises(ValueError, match="cannot be negative"):
        build_customer_features(valid_customer)


def test_invalid_senior_citizen_value_is_rejected(valid_customer):
    valid_customer["SeniorCitizen"] = 2

    with pytest.raises(ValueError, match="SeniorCitizen"):
        build_customer_features(valid_customer)


def test_prediction_returns_expected_structure(valid_customer):
    result = predict(valid_customer)

    assert set(result.keys()) == {
        "prediction",
        "churn_probability",
        "threshold",
    }

    assert result["prediction"] in [
        "Likely to churn",
        "Not likely to churn",
    ]

    assert isinstance(result["churn_probability"], float)
    assert 0 <= result["churn_probability"] <= 1
    assert result["threshold"] == CHURN_THRESHOLD


def test_prediction_uses_configured_threshold(valid_customer):
    result = predict(valid_customer)

    if result["churn_probability"] >= CHURN_THRESHOLD:
        assert result["prediction"] == "Likely to churn"
    else:
        assert result["prediction"] == "Not likely to churn"
