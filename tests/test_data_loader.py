"""
Tests for Data Loader Module
Covers Scenarios:
1. Valid CSV upload
2. Empty CSV upload
6. Different column names
7. CSV files containing special characters
"""

import io
import pytest
import pandas as pd
from modules.data_loader import load_csv, sanitize_column_names, get_schema_metadata, DataLoaderError


def test_scenario_1_valid_csv_upload():
    """Scenario 1: Valid CSV upload."""
    csv_data = "date,region,sales\n2024-01-01,North,100\n2024-01-02,South,200\n"
    df, mapping = load_csv(csv_data)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "region" in df.columns
    assert "sales" in df.columns
    assert df["sales"].iloc[0] == 100


def test_scenario_2_empty_csv_upload():
    """Scenario 2: Empty CSV upload."""
    empty_csv = ""
    with pytest.raises(DataLoaderError, match="empty|no columns"):
        load_csv(empty_csv)

    empty_buffer = io.BytesIO(b"")
    with pytest.raises(DataLoaderError, match="empty|no columns"):
        load_csv(empty_buffer)


def test_scenario_6_different_column_names():
    """Scenario 6: Different column names (arbitrary datasets)."""
    custom_csv = (
        "employee_id,department_name,hourly_wage,is_manager\n"
        "E101,Engineering,65.50,True\n"
        "E102,Marketing,45.00,False\n"
    )
    df, mapping = load_csv(custom_csv)
    assert set(df.columns) == {"employee_id", "department_name", "hourly_wage", "is_manager"}
    assert df["hourly_wage"].dtype in ["float64", "float32"]


def test_scenario_7_special_characters_in_columns():
    """Scenario 7: CSV files containing special characters, whitespace, and duplicates in headers."""
    messy_cols = [" Total Sales ($) ", "Region / Territory", "Product % Share", "sales", "sales"]
    sanitized, mapping = sanitize_column_names(messy_cols)

    # Asserts that spaces and symbols are replaced with clean underscores
    assert sanitized[0] == "total_sales"
    assert sanitized[1] == "region_territory"
    assert sanitized[2] == "product_share"
    assert sanitized[3] == "sales"
    assert sanitized[4] == "sales_1"  # Deduplicated suffix


def test_schema_metadata_extraction():
    """Verifies schema metadata extraction generates types and sample values."""
    csv_data = "id,category,score\n1,Alpha,95.5\n2,Beta,88.0\n"
    df, _ = load_csv(csv_data)
    meta = get_schema_metadata(df)

    assert meta["total_rows"] == 2
    assert meta["total_columns"] == 3
    assert meta["columns"]["category"]["sqlite_type"] == "TEXT"
    assert "Alpha" in meta["columns"]["category"]["sample_values"]
