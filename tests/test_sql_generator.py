"""
Tests for SQL Generator Module
Covers Scenarios:
8. Valid natural-language questions
9. Invalid or unrelated questions
10. SQL generation errors
"""

from unittest.mock import MagicMock
import pytest
from modules.sql_generator import SQLGenerator, SQLGeneratorError
from modules.llm_service import GroqLLMService


@pytest.fixture
def sample_metadata():
    return {
        "total_rows": 100,
        "total_columns": 4,
        "columns": {
            "date": {"sqlite_type": "TIMESTAMP", "sample_values": ["2024-01-01"]},
            "region": {"sqlite_type": "TEXT", "sample_values": ["North", "South"]},
            "sales": {"sqlite_type": "REAL", "sample_values": ["1200.50", "800.00"]},
            "quantity": {"sqlite_type": "INTEGER", "sample_values": ["2", "5"]}
        }
    }


def test_scenario_8_valid_natural_language_question(sample_metadata):
    """Scenario 8: Valid question prompt incorporates table schema, types, and sample values."""
    mock_llm = MagicMock(spec=GroqLLMService)
    mock_llm.generate_chat_completion.return_value = "SELECT region, SUM(sales) FROM data_table GROUP BY region;"

    generator = SQLGenerator(llm_service=mock_llm)
    ddl = "CREATE TABLE data_table (date TIMESTAMP, region TEXT, sales REAL, quantity INTEGER);"
    sql = generator.generate_sql("What is the total sales by region?", "data_table", ddl, sample_metadata)

    assert "SELECT region, SUM(sales)" in sql
    # Verify the LLM was called with the system prompt containing sample values and DDL
    call_args = mock_llm.generate_chat_completion.call_args[0][0]
    system_prompt = call_args[0]["content"]
    assert "data_table" in system_prompt
    assert "North" in system_prompt


def test_scenario_9_unrelated_questions_handling(sample_metadata):
    """Scenario 9: Unrelated questions that cannot be answered return unanswerable marker."""
    mock_llm = MagicMock(spec=GroqLLMService)
    mock_llm.generate_chat_completion.return_value = "-- UNANSWERABLE: The dataset does not contain weather or temperature information."

    generator = SQLGenerator(llm_service=mock_llm)
    ddl = "CREATE TABLE data_table (date TIMESTAMP, region TEXT, sales REAL, quantity INTEGER);"
    raw_sql = generator.generate_sql("What was the weather yesterday?", "data_table", ddl, sample_metadata)

    assert "UNANSWERABLE" in raw_sql


def test_scenario_10_sql_generation_errors():
    """Scenario 10: Empty or invalid input questions raise SQLGeneratorError."""
    mock_llm = MagicMock(spec=GroqLLMService)
    generator = SQLGenerator(llm_service=mock_llm)

    with pytest.raises(SQLGeneratorError, match="Question cannot be empty"):
        generator.generate_sql("", "data_table", "", {})

    with pytest.raises(SQLGeneratorError, match="Question cannot be empty"):
        generator.generate_sql("   ", "data_table", "", {})


def test_dynamic_suggested_questions(sample_metadata):
    """Verifies suggested questions are synthesized dynamically from columns."""
    suggestions = SQLGenerator.generate_suggested_questions(sample_metadata)
    assert len(suggestions) > 0
    # Date + Numeric should produce a time trend question
    has_trend = any("trend" in s.lower() or "over time" in s.lower() for s in suggestions)
    assert has_trend
    # Categorical + Numeric should produce a group by question
    has_cat = any("sales by region" in s.lower() or "region" in s.lower() for s in suggestions)
    assert has_cat
