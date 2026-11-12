"""
Data validation for the Telco Customer Churn dataset (Great Expectations 1.x API).

Great Expectations 1.0 removed the old `ge.dataset.PandasDataset` API, so this
version uses an ephemeral context + batch + expectation classes instead.
Same checks as before, same return value: (is_valid, list_of_failed_expectation_types).
"""
from typing import List, Tuple

import pandas as pd
import great_expectations as gx
import great_expectations.expectations as gxe


def _build_expectations() -> list:
    exp = []

    # --- Schema: essential columns must exist ---
    for col in [
        "customerID", "gender", "Partner", "Dependents", "PhoneService",
        "InternetService", "Contract", "tenure", "MonthlyCharges", "TotalCharges",
    ]:
        exp.append(gxe.ExpectColumnToExist(column=col))
    exp.append(gxe.ExpectColumnValuesToNotBeNull(column="customerID"))

    # --- Business logic: allowed categorical values ---
    exp.append(gxe.ExpectColumnValuesToBeInSet(column="gender", value_set=["Male", "Female"]))
    for col in ["Partner", "Dependents", "PhoneService"]:
        exp.append(gxe.ExpectColumnValuesToBeInSet(column=col, value_set=["Yes", "No"]))
    exp.append(gxe.ExpectColumnValuesToBeInSet(
        column="Contract", value_set=["Month-to-month", "One year", "Two year"]))
    exp.append(gxe.ExpectColumnValuesToBeInSet(
        column="InternetService", value_set=["DSL", "Fiber optic", "No"]))

    # --- Numeric ranges ---
    exp.append(gxe.ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120))
    exp.append(gxe.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0, max_value=200))
    exp.append(gxe.ExpectColumnValuesToBeBetween(column="TotalCharges", min_value=0))

    # --- No missing values in critical numeric features ---
    exp.append(gxe.ExpectColumnValuesToNotBeNull(column="tenure"))
    exp.append(gxe.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"))

    # --- Consistency: TotalCharges >= MonthlyCharges (5% tolerance for new customers) ---
    exp.append(gxe.ExpectColumnPairValuesAToBeGreaterThanB(
        column_A="TotalCharges", column_B="MonthlyCharges", or_equal=True, mostly=0.95))

    return exp


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    print("🔍 Starting data validation with Great Expectations...")

    # The raw CSV stores TotalCharges as text and has 11 blank values (" ").
    # Coerce on a COPY so the numeric checks work; preprocessing does the same later.
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("telco_source")
    asset = data_source.add_dataframe_asset(name="telco_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("telco_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name="telco_suite")
    for e in _build_expectations():
        suite.add_expectation(e)

    results = batch.validate(suite)

    failed: List[str] = []
    for r in results.results:
        if not r.success:
            failed.append(r.expectation_config.type)

    total = len(results.results)
    passed = total - len(failed)
    if results.success:
        print(f"✅ Data validation PASSED: {passed}/{total} checks successful")
    else:
        print(f"❌ Data validation FAILED: {len(failed)}/{total} checks failed")
        print(f"   Failed expectations: {failed}")

    return bool(results.success), failed