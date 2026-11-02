"""
Comprehensive End-to-End Test Suite
Directly validates all 20 scenarios specified in Section 13 of the project brief.
All LLM interactions use mocked responses for 100% deterministic, offline test execution.
"""

from unittest.mock import MagicMock
import io
import sqlite3
import pytest
import pandas as pd
import numpy as np

from modules.data_loader import load_csv, sanitize_column_names, DataLoaderError
from modules.data_cleaner import DataCleaner
from modules.database import SQLiteManager
from modules.llm_service import GroqLLMService, LLMServiceError
from modules.sql_generator import SQLGenerator, SQLGeneratorError
from modules.sql_validator import clean_sql_output, validate_sql, SQLValidationError, UnanswerableQuestionError
from modules.query_executor import QueryExecutor, QueryExecutionError
from modules.visualizer import ChartSelector
from modules.insights import InsightsGenerator


# Scenario 1: Valid CSV upload
def test_scenario_01_valid_csv_upload():
    csv_text = "id,name,value\n1,Alpha,10.5\n2,Beta,20.0\n"
    df, mapping = load_csv(csv_text)
    assert len(df) == 2
    assert list(df.columns) == ["id", "name", "value"]


# Scenario 2: Empty CSV upload
def test_scenario_02_empty_csv_upload():
    with pytest.raises(DataLoaderError):
        load_csv("")


# Scenario 3: Missing values
def test_scenario_03_missing_values():
    df = pd.DataFrame({"col": [1.0, np.nan, 3.0]})
    cleaner = DataCleaner(df)
    report = cleaner.generate_quality_report()
    assert report["missing_stats"]["col"]["missing_count"] == 1
    cleaner.handle_missing_values("col", strategy="mean")
    assert cleaner.get_cleaned_dataframe()["col"].isna().sum() == 0


# Scenario 4: Duplicate records
def test_scenario_04_duplicate_records():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]})
    cleaner = DataCleaner(df)
    assert cleaner.remove_duplicates() == 1
    assert len(cleaner.get_cleaned_dataframe()) == 2


# Scenario 5: Invalid numeric values
def test_scenario_05_invalid_numeric_values():
    df = pd.DataFrame({"num": ["100", "invalid_val", "$250.00"]})
    cleaner = DataCleaner(df)
    valid, coerced = cleaner.convert_numeric("num")
    assert valid == 2
    assert coerced == 1
    assert pd.isna(cleaner.get_cleaned_dataframe().loc[1, "num"])


# Scenario 6: Different column names
def test_scenario_06_different_column_names():
    df, _ = load_csv("foo_bar,baz_qux\n1,hello\n")
    assert list(df.columns) == ["foo_bar", "baz_qux"]


# Scenario 7: CSV files containing special characters
def test_scenario_07_special_characters_in_columns():
    cols, _ = sanitize_column_names([" Order # ", "Sales (USD)", "Customer %"])
    assert cols == ["order", "sales_usd", "customer"]


# Scenario 8: Valid natural-language questions
def test_scenario_08_valid_natural_language_question():
    mock_llm = MagicMock(spec=GroqLLMService)
    mock_llm.generate_chat_completion.return_value = "SELECT region, SUM(sales) FROM data_table GROUP BY region;"
    generator = SQLGenerator(llm_service=mock_llm)
    sql = generator.generate_sql("Total sales by region?", "data_table", "CREATE TABLE ...", {})
    assert "SELECT region" in sql


# Scenario 9: Invalid or unrelated questions
def test_scenario_09_invalid_or_unrelated_questions():
    mock_llm = MagicMock(spec=GroqLLMService)
    mock_llm.generate_chat_completion.return_value = "-- UNANSWERABLE: Dataset lacks population stats."
    generator = SQLGenerator(llm_service=mock_llm)
    sql = generator.generate_sql("What is the population of Tokyo?", "data_table", "CREATE TABLE ...", {})
    with pytest.raises(UnanswerableQuestionError):
        clean_sql_output(sql)


# Scenario 10: SQL generation errors
def test_scenario_10_sql_generation_errors():
    generator = SQLGenerator(llm_service=MagicMock())
    with pytest.raises(SQLGeneratorError):
        generator.generate_sql("", "data_table", "", {})


# Scenario 11: Markdown-formatted SQL output
def test_scenario_11_markdown_formatted_sql():
    raw_markdown = "```sql\nSELECT * FROM data_table\n```"
    cleaned = clean_sql_output(raw_markdown)
    assert cleaned == "SELECT * FROM data_table"


# Scenario 12: SQL syntax errors
def test_scenario_12_sql_syntax_errors():
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT FROM WHERE data_table", allowed_tables=["data_table"])


# Scenario 13: Unsafe SQL statements
def test_scenario_13_unsafe_sql_statements():
    with pytest.raises(SQLValidationError):
        validate_sql("DROP TABLE data_table", allowed_tables=["data_table"])
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT * FROM data_table; DELETE FROM data_table", allowed_tables=["data_table"])


# Scenario 14: Empty query results
def test_scenario_14_empty_query_results():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE data_table (id INT);")
    executor = QueryExecutor(conn)
    df, _, count, _ = executor.execute_query("SELECT * FROM data_table WHERE id = 999;")
    assert len(df) == 0
    assert count == 0
    conn.close()


# Scenario 15: Groq API errors
def test_scenario_15_groq_api_errors():
    service = GroqLLMService(api_key="mock_key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("429 rate limit reached")
    service._client = mock_client
    with pytest.raises(LLMServiceError, match="Rate Limit Exceeded"):
        service.generate_chat_completion([{"role": "user", "content": "hi"}])


# Scenario 16: Groq model unavailability
def test_scenario_16_groq_model_unavailability():
    service = GroqLLMService(api_key="mock_key", model="deprecated-model")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("model_not_found: model is decommissioned")
    service._client = mock_client
    with pytest.raises(LLMServiceError, match="Model Error"):
        service.generate_chat_completion([{"role": "user", "content": "hi"}])


# Scenario 17: Chart generation for suitable results
def test_scenario_17_chart_generation_suitable_results():
    df = pd.DataFrame({"region": ["A", "B"], "sales": [10, 20]})
    chart_type, x, y, _ = ChartSelector.recommend_chart_type(df)
    assert chart_type in ["pie", "bar"]
    fig = ChartSelector.render_figure(df, chart_type, x, y)
    assert fig is not None


# Scenario 18: Queries that cannot be visualized
def test_scenario_18_queries_cannot_be_visualized():
    df = pd.DataFrame({"text_info": ["Row 1", "Row 2"]})
    chart_type, _, _, rationale = ChartSelector.recommend_chart_type(df)
    assert chart_type == "table"


# Scenario 19: Large result sets
def test_scenario_19_large_result_sets():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE data_table (id INT);")
    conn.executemany("INSERT INTO data_table VALUES (?);", [(i,) for i in range(20)])
    executor = QueryExecutor(conn, max_rows=5)
    df, _, count, truncated = executor.execute_query("SELECT * FROM data_table;")
    assert len(df) == 5
    assert truncated is True
    conn.close()


# Scenario 20: Missing API configuration
def test_scenario_20_missing_api_configuration():
    service = GroqLLMService(api_key="", mock_mode=False)
    with pytest.raises(LLMServiceError, match="Missing Groq API Key"):
        service.generate_chat_completion([{"role": "user", "content": "hi"}])
