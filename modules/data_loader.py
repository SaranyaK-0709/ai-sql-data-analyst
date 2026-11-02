"""
Data Loader Module
Handles robust CSV loading, encoding resolution, column sanitization, and schema introspection.
"""

from typing import Union, BinaryIO, TextIO, Tuple, Dict, Any, List
import io
import os
import re
import pandas as pd


class DataLoaderError(Exception):
    """Custom exception raised when CSV loading or parsing fails."""
    pass


def sanitize_column_names(columns: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Sanitizes column names for safe SQL and DataFrame manipulation.
    - Trims whitespace and removes invisible control characters.
    - Replaces internal spaces and special characters with underscores if needed,
      or preserves clean readable names while avoiding SQL keywords collision.
    - Handles duplicated column names gracefully by appending an index suffix.

    Returns:
        Tuple of (sanitized_columns_list, mapping_from_sanitized_to_original)
    """
    sanitized: List[str] = []
    mapping: Dict[str, str] = {}
    seen_counts: Dict[str, int] = {}

    for col in columns:
        col_str = str(col).strip()
        if not col_str:
            base_name = "unnamed_column"
        else:
            # Replace whitespace and problematic characters with underscore
            # Keep letters, numbers, and underscores
            clean_name = re.sub(r"[^\w\s]", "", col_str)
            clean_name = re.sub(r"\s+", "_", clean_name).strip("_").lower()
            base_name = clean_name if clean_name else "col"

        # Handle duplicates
        count = seen_counts.get(base_name, 0)
        if count == 0:
            final_name = base_name
            seen_counts[base_name] = 1
        else:
            final_name = f"{base_name}_{count}"
            seen_counts[base_name] = count + 1

        sanitized.append(final_name)
        mapping[final_name] = col_str

    return sanitized, mapping


def load_csv(file_source: Union[str, BinaryIO, TextIO, bytes]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Loads a CSV dataset from a file path, file-like buffer, or raw bytes.
    Attempts UTF-8 first, with automatic fallback to latin1 / cp1252.

    Returns:
        Tuple of (loaded_dataframe, column_name_mapping)

    Raises:
        DataLoaderError: If the file is empty, malformed, or cannot be parsed.
    """
    if file_source is None:
        raise DataLoaderError("No file or data stream provided.")

    if isinstance(file_source, str) and not file_source.strip():
        raise DataLoaderError("The provided CSV file is completely empty.")
    if isinstance(file_source, bytes) and not file_source.strip():
        raise DataLoaderError("The provided CSV file is completely empty.")

    encodings = ["utf-8", "latin-1", "cp1252"]
    df = None
    last_error = None

    for enc in encodings:
        try:
            if isinstance(file_source, bytes):
                buffer = io.BytesIO(file_source)
                df = pd.read_csv(buffer, encoding=enc)
            elif isinstance(file_source, str):
                if "\n" in file_source or "\r" in file_source or not os.path.exists(file_source):
                    buffer = io.StringIO(file_source)
                    df = pd.read_csv(buffer, encoding=enc)
                else:
                    df = pd.read_csv(file_source, encoding=enc)
            elif hasattr(file_source, "seek"):
                file_source.seek(0)
                df = pd.read_csv(file_source, encoding=enc)
            else:
                df = pd.read_csv(file_source, encoding=enc)
            break
        except UnicodeDecodeError as ude:
            last_error = ude
            continue
        except pd.errors.EmptyDataError:
            raise DataLoaderError("The provided CSV file is completely empty.")
        except Exception as exc:
            last_error = exc
            continue

    if df is None:
        raise DataLoaderError(f"Failed to load CSV file. Error: {str(last_error)}")

    if df.empty and len(df.columns) == 0:
        raise DataLoaderError("The CSV file contains no columns or rows.")

    if len(df.columns) == 0:
        raise DataLoaderError("No header or columns detected in CSV.")

    # Sanitize columns
    clean_cols, mapping = sanitize_column_names(df.columns.tolist())
    df.columns = clean_cols

    return df, mapping


def get_schema_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts detailed schema metadata from a DataFrame.
    Returns column names, types, non-null counts, null percentages, and sample values.
    """
    metadata: Dict[str, Any] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": {}
    }

    for col in df.columns:
        series = df[col]
        non_null_count = int(series.count())
        null_count = int(series.isna().sum())
        null_percentage = round((null_count / len(df)) * 100, 2) if len(df) > 0 else 0.0

        # Extract up to 3 non-null sample unique values for LLM context
        sample_vals = series.dropna().unique()[:3].tolist()
        sample_vals_repr = [str(v) for v in sample_vals]

        metadata["columns"][col] = {
            "pandas_dtype": str(series.dtype),
            "sqlite_type": map_dtype_to_sqlite(series.dtype),
            "non_null_count": non_null_count,
            "null_count": null_count,
            "null_percentage": null_percentage,
            "unique_values_count": int(series.nunique()),
            "sample_values": sample_vals_repr
        }

    return metadata


def map_dtype_to_sqlite(dtype: Any) -> str:
    """Maps a Pandas data type to its SQLite column type affinity."""
    dtype_str = str(dtype).lower()
    if "int" in dtype_str:
        return "INTEGER"
    if "float" in dtype_str or "decimal" in dtype_str:
        return "REAL"
    if "bool" in dtype_str:
        return "INTEGER"
    if "datetime" in dtype_str or "timestamp" in dtype_str:
        return "TIMESTAMP"
    return "TEXT"
