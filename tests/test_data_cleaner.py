"""
Tests for Data Cleaner Module
Covers Scenarios:
3. Missing values
4. Duplicate records
5. Invalid numeric values
"""

import pytest
import pandas as pd
import numpy as np
from modules.data_cleaner import DataCleaner, DataCleanerError


def test_scenario_3_missing_values():
    """Scenario 3: Missing values detection, reporting, and controlled imputation."""
    data = {
        "region": ["North", "South", None, "North"],
        "sales": [100.0, np.nan, 300.0, 400.0]
    }
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    report = cleaner.generate_quality_report()

    assert report["missing_stats"]["region"]["missing_count"] == 1
    assert report["missing_stats"]["sales"]["missing_count"] == 1

    # Impute sales with mean
    cleaner.handle_missing_values("sales", strategy="mean")
    cleaned_df = cleaner.get_cleaned_dataframe()
    assert cleaned_df["sales"].isna().sum() == 0
    # Mean of 100, 300, 400 is 266.67
    assert round(cleaned_df.loc[1, "sales"], 2) == 266.67

    # Drop missing in region
    cleaner.handle_missing_values("region", strategy="drop")
    cleaned_df = cleaner.get_cleaned_dataframe()
    assert len(cleaned_df) == 3
    assert cleaned_df["region"].isna().sum() == 0


def test_scenario_4_duplicate_records():
    """Scenario 4: Duplicate records detection and removal."""
    data = {
        "id": [1, 2, 2, 3],
        "name": ["Alice", "Bob", "Bob", "Charlie"]
    }
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    report = cleaner.generate_quality_report()

    assert report["duplicate_rows_count"] == 1

    removed = cleaner.remove_duplicates()
    assert removed == 1
    assert len(cleaner.get_cleaned_dataframe()) == 3


def test_scenario_5_invalid_numeric_values():
    """Scenario 5: Invalid numeric values are coerced safely without inventing numbers."""
    data = {
        "product": ["Phone", "Laptop", "Tablet", "Monitor"],
        "price": ["$500", "abc", "1,200.50", "300"]
    }
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)

    valid_cnt, coerced_cnt = cleaner.convert_numeric("price")
    cleaned_df = cleaner.get_cleaned_dataframe()

    # '$500', '1,200.50', '300' are converted to 500, 1200.5, 300
    # 'abc' is safely coerced to NaN
    assert valid_cnt == 3
    assert coerced_cnt == 1
    assert pd.isna(cleaned_df.loc[1, "price"])
    assert cleaned_df.loc[0, "price"] == 500.0
    assert cleaned_df.loc[2, "price"] == 1200.5


def test_capitalization_and_whitespace_cleaning():
    """Verifies whitespace trimming and capitalization standardization."""
    data = {
        "region": [" north ", "NORTH", "North ", "South"]
    }
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)

    # Trim whitespace
    affected_ws = cleaner.trim_whitespace(["region"])
    assert affected_ws >= 2

    # Standardize casing to Title Case
    cleaner.standardize_capitalization("region", style="title")
    cleaned_df = cleaner.get_cleaned_dataframe()

    # All 'north' variations should now be 'North'
    north_entries = cleaned_df[cleaned_df["region"] == "North"]
    assert len(north_entries) == 3
