
import pandas as pd

from src.features.build_features import (
    _map_binary_series,
    build_features,
)


def test_yes_no_binary_mapping():
    series = pd.Series(["No", "Yes", "Yes", "No"])

    result = _map_binary_series(series)

    assert result.tolist() == [0, 1, 1, 0]


def test_gender_binary_mapping():
    series = pd.Series(["Female", "Male", "Female"])

    result = _map_binary_series(series)

    assert result.tolist() == [0, 1, 0]


def test_generic_binary_mapping():
    series = pd.Series(["Beta", "Alpha", "Beta"])

    result = _map_binary_series(series)

    assert result.tolist() == [1, 0, 1]


def test_build_features_encodes_binary_columns():
    df = pd.DataFrame({
        "gender": ["Female", "Male", "Female"],
        "Partner": ["Yes", "No", "Yes"],
        "tenure": [5, 12, 24],
        "Churn": ["No", "Yes", "No"],
    })

    result = build_features(df)

    assert result["gender"].tolist() == [0, 1, 0]
    assert result["Partner"].tolist() == [1, 0, 1]

    # Target must remain unchanged.
    assert result["Churn"].tolist() == ["No", "Yes", "No"]


def test_build_features_one_hot_encodes_categories():
    df = pd.DataFrame({
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year",
        ],
        "Churn": ["No", "Yes", "No"],
    })

    result = build_features(df)

    assert "Contract" not in result.columns
    assert "Contract_One year" in result.columns
    assert "Contract_Two year" in result.columns

    assert result["Contract_One year"].tolist() == [0, 1, 0]
    assert result["Contract_Two year"].tolist() == [0, 0, 1]


def test_build_features_preserves_original_dataframe():
    df = pd.DataFrame({
        "Partner": ["Yes", "No"],
        "tenure": [5, 12],
        "Churn": ["No", "Yes"],
    })

    original = df.copy(deep=True)

    build_features(df)

    pd.testing.assert_frame_equal(df, original)


def test_build_features_converts_boolean_columns():
    df = pd.DataFrame({
        "IsActive": [True, False, True],
        "tenure": [5, 12, 24],
        "Churn": ["No", "Yes", "No"],
    })

    result = build_features(df)

    assert result["IsActive"].tolist() == [1, 0, 1]
    assert pd.api.types.is_integer_dtype(result["IsActive"])
