"""
Data Cleaner Module
Performs comprehensive data-quality analysis and controlled, auditable cleaning operations.
"""

from typing import Dict, Any, List, Optional, Tuple
import io
import re
import pandas as pd
import numpy as np


class DataCleanerError(Exception):
    """Custom exception raised during data cleaning operations."""
    pass


class DataCleaner:
    """
    Audits dataset quality and executes user-directed cleaning transformations
    with full traceability and before/after comparisons.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        if df is None:
            self.original_df = pd.DataFrame()
            self.current_df = pd.DataFrame()
        else:
            self.original_df = df.copy()
            self.current_df = df.copy()
        self.changelog: List[str] = []

    def generate_quality_report(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generates a comprehensive data-quality health report.
        """
        target_df = self.current_df if df is None else df
        total_rows = len(target_df)
        total_cols = len(target_df.columns)

        # 1. Duplicate rows
        duplicate_rows_count = int(target_df.duplicated().sum())

        # 2. Missing values per column
        missing_stats: Dict[str, Dict[str, Any]] = {}
        for col in target_df.columns:
            null_count = int(target_df[col].isna().sum())
            null_pct = round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
            missing_stats[col] = {
                "missing_count": null_count,
                "missing_percentage": null_pct
            }

        # 3. Completely empty rows / columns
        empty_rows_count = int(target_df.isna().all(axis=1).sum())
        empty_cols = [col for col in target_df.columns if target_df[col].isna().all()]

        # 4. Whitespace anomalies in string columns
        whitespace_issues: Dict[str, int] = {}
        # 5. Capitalization inconsistencies in text columns
        capitalization_issues: Dict[str, List[str]] = {}
        # 6. Potential numeric columns stored as text
        numeric_candidates: List[str] = []
        # 7. Potential date columns stored as text
        date_candidates: List[str] = []

        for col in target_df.columns:
            series = target_df[col]
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                non_null_str = series.dropna().astype(str)
                if len(non_null_str) == 0:
                    continue

                # Whitespace check
                has_whitespace = non_null_str.apply(lambda s: s != s.strip())
                ws_count = int(has_whitespace.sum())
                if ws_count > 0:
                    whitespace_issues[col] = ws_count

                # Capitalization check: groups of unique values that differ only by case
                unique_vals = non_null_str.unique()
                lower_map: Dict[str, List[str]] = {}
                for val in unique_vals:
                    val_clean = val.strip()
                    lower_key = val_clean.lower()
                    lower_map.setdefault(lower_key, []).append(val_clean)

                clashing_groups = [group for group in lower_map.values() if len(set(group)) > 1]
                if clashing_groups:
                    # Flatten up to 3 groups
                    examples = []
                    for g in clashing_groups[:3]:
                        examples.append(" / ".join(list(set(g))))
                    capitalization_issues[col] = examples

                # Potential numeric check: e.g. currency '$100', commas '1,000', or mixed '123'
                cleaned_num = non_null_str.str.replace(r"[$,€£¥\s]", "", regex=True)
                numeric_converted = pd.to_numeric(cleaned_num, errors="coerce")
                valid_num_pct = numeric_converted.notna().sum() / len(non_null_str)
                if valid_num_pct > 0.70 and not pd.api.types.is_numeric_dtype(series):
                    numeric_candidates.append(col)

                # Potential date check:
                sample_str = non_null_str.head(20)
                date_count = 0
                for item in sample_str:
                    if re.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", item) or re.match(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}", item):
                        date_count += 1
                if date_count / len(sample_str) > 0.5:
                    date_candidates.append(col)

        quality_score = 100
        if duplicate_rows_count > 0:
            quality_score -= min(20, int((duplicate_rows_count / max(1, total_rows)) * 100))
        total_missing = sum(item["missing_count"] for item in missing_stats.values())
        if total_missing > 0:
            quality_score -= min(30, int((total_missing / max(1, total_rows * total_cols)) * 100))
        if whitespace_issues:
            quality_score -= 10
        if capitalization_issues:
            quality_score -= 10

        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "duplicate_rows_count": duplicate_rows_count,
            "missing_stats": missing_stats,
            "empty_rows_count": empty_rows_count,
            "empty_columns": empty_cols,
            "whitespace_issues": whitespace_issues,
            "capitalization_issues": capitalization_issues,
            "numeric_candidates": numeric_candidates,
            "date_candidates": date_candidates,
            "quality_score": max(0, quality_score),
            "changelog": self.changelog
        }

    def remove_duplicates(self) -> int:
        """Removes duplicate rows. Returns count of removed rows."""
        before = len(self.current_df)
        self.current_df.drop_duplicates(inplace=True)
        self.current_df.reset_index(drop=True, inplace=True)
        removed = before - len(self.current_df)
        if removed > 0:
            self.changelog.append(f"Removed {removed} duplicate rows.")
        return removed

    def trim_whitespace(self, columns: Optional[List[str]] = None) -> int:
        """Trims leading and trailing whitespace from string columns."""
        target_cols = columns if columns else self.current_df.select_dtypes(include=["object", "string"]).columns.tolist()
        affected = 0
        for col in target_cols:
            if col in self.current_df.columns:
                series = self.current_df[col]
                if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                    trimmed = series.astype(str).str.strip()
                    # preserve actual NaNs
                    trimmed = trimmed.where(series.notna(), np.nan)
                    diff_count = int((series != trimmed).sum())
                    if diff_count > 0:
                        affected += diff_count
                        self.current_df[col] = trimmed
                        self.changelog.append(f"Trimmed whitespace on column '{col}' ({diff_count} cells).")
        return affected

    def standardize_capitalization(self, column: str, style: str = "title") -> int:
        """
        Standardizes text capitalization for a specific column.
        style can be 'title', 'upper', or 'lower'.
        """
        if column not in self.current_df.columns:
            return 0

        series = self.current_df[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            return 0

        if style == "title":
            transformed = series.astype(str).str.title()
        elif style == "upper":
            transformed = series.astype(str).str.upper()
        elif style == "lower":
            transformed = series.astype(str).str.lower()
        else:
            raise DataCleanerError(f"Unsupported capitalization style '{style}'.")

        transformed = transformed.where(series.notna(), np.nan)
        diff_count = int((series != transformed).sum())
        if diff_count > 0:
            self.current_df[column] = transformed
            self.changelog.append(f"Standardized '{column}' to {style} case ({diff_count} cells).")
        return diff_count

    def convert_numeric(self, column: str) -> Tuple[int, int]:
        """
        Converts a column to numeric, removing currency symbols, commas, or spaces.
        Invalid strings are safely coerced to NaN (never invented).
        Returns:
            Tuple of (successfully_converted_count, invalid_coerced_to_nan_count)
        """
        if column not in self.current_df.columns:
            return 0, 0

        series = self.current_df[column]
        if pd.api.types.is_numeric_dtype(series):
            return len(series), 0

        # Clean common formatting like $, commas, spaces
        cleaned = series.astype(str).str.replace(r"[$,€£¥\s]", "", regex=True)
        converted = pd.to_numeric(cleaned, errors="coerce")
        # Where original was non-null but converted is NaN -> invalid values
        invalid_count = int((series.notna() & converted.isna()).sum())
        valid_count = int(converted.notna().sum())

        self.current_df[column] = converted
        self.changelog.append(f"Converted '{column}' to numeric ({valid_count} valid, {invalid_count} coerced to NaN).")
        return valid_count, invalid_count

    def convert_datetime(self, column: str) -> int:
        """Converts a column to datetime format."""
        if column not in self.current_df.columns:
            return 0

        series = self.current_df[column]
        converted = pd.to_datetime(series, errors="coerce")
        valid_count = int(converted.notna().sum())
        self.current_df[column] = converted
        self.changelog.append(f"Converted '{column}' to datetime ({valid_count} valid timestamps).")
        return valid_count

    def handle_missing_values(self, column: str, strategy: str = "drop", fill_value: Any = None) -> int:
        """
        Handles missing values in a column.
        strategy: 'drop' | 'mean' | 'median' | 'mode' | 'constant'
        """
        if column not in self.current_df.columns:
            return 0

        null_count = int(self.current_df[column].isna().sum())
        if null_count == 0:
            return 0

        if strategy == "drop":
            before = len(self.current_df)
            self.current_df.dropna(subset=[column], inplace=True)
            self.current_df.reset_index(drop=True, inplace=True)
            dropped = before - len(self.current_df)
            self.changelog.append(f"Dropped {dropped} rows with missing values in '{column}'.")
            return dropped

        elif strategy == "mean":
            if not pd.api.types.is_numeric_dtype(self.current_df[column]):
                raise DataCleanerError(f"Mean imputation is only valid for numeric columns. '{column}' is {self.current_df[column].dtype}.")
            mean_val = self.current_df[column].mean()
            self.current_df[column] = self.current_df[column].fillna(mean_val)
            self.changelog.append(f"Filled {null_count} missing values in '{column}' with mean ({round(mean_val, 2)}).")
            return null_count

        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(self.current_df[column]):
                raise DataCleanerError(f"Median imputation is only valid for numeric columns. '{column}' is {self.current_df[column].dtype}.")
            median_val = self.current_df[column].median()
            self.current_df[column] = self.current_df[column].fillna(median_val)
            self.changelog.append(f"Filled {null_count} missing values in '{column}' with median ({round(median_val, 2)}).")
            return null_count

        elif strategy == "mode":
            mode_series = self.current_df[column].mode()
            mode_val = mode_series[0] if len(mode_series) > 0 else "Unknown"
            self.current_df[column] = self.current_df[column].fillna(mode_val)
            self.changelog.append(f"Filled {null_count} missing values in '{column}' with mode ('{mode_val}').")
            return null_count

        elif strategy == "constant":
            val = fill_value if fill_value is not None else "Unknown"
            self.current_df[column] = self.current_df[column].fillna(val)
            self.changelog.append(f"Filled {null_count} missing values in '{column}' with constant ('{val}').")
            return null_count

        else:
            raise DataCleanerError(f"Unsupported missing value strategy: '{strategy}'.")

    def drop_empty_rows_and_columns(self) -> Tuple[int, int]:
        """Drops rows and columns that are 100% empty (NaN)."""
        before_rows = len(self.current_df)
        before_cols = len(self.current_df.columns)

        self.current_df.dropna(how="all", inplace=True)
        self.current_df.reset_index(drop=True, inplace=True)
        dropped_rows = before_rows - len(self.current_df)

        self.current_df.dropna(axis=1, how="all", inplace=True)
        dropped_cols = before_cols - len(self.current_df.columns)

        if dropped_rows > 0:
            self.changelog.append(f"Dropped {dropped_rows} completely empty rows.")
        if dropped_cols > 0:
            self.changelog.append(f"Dropped {dropped_cols} completely empty columns.")

        return dropped_rows, dropped_cols

    def get_cleaned_dataframe(self) -> pd.DataFrame:
        """Returns the current cleaned DataFrame."""
        return self.current_df

    def to_csv_bytes(self) -> bytes:
        """Encodes the cleaned dataset into CSV bytes for user download."""
        buffer = io.StringIO()
        self.current_df.to_csv(buffer, index=False)
        return buffer.getvalue().encode("utf-8")
