"""
Tests for SQL Validator Module
Covers Scenarios:
11. Markdown-formatted SQL output
12. SQL syntax errors
13. Unsafe SQL statements
"""

import pytest
from modules.sql_validator import (
    clean_sql_output,
    validate_sql,
    SQLValidationError,
    UnanswerableQuestionError
)


def test_scenario_11_markdown_formatted_sql():
    """Scenario 11: Markdown code fences and preambles are stripped cleanly."""
    raw_response_with_fence = """Here is the SQL query for your question:
```sql
SELECT region, SUM(sales) AS total_sales
FROM data_table
GROUP BY region;
```
Hope this helps!"""

    cleaned = clean_sql_output(raw_response_with_fence)
    assert not cleaned.startswith("```")
    assert not cleaned.endswith("```")
    assert "Here is the SQL" not in cleaned
    assert "Hope this helps" not in cleaned
    assert "SELECT region, SUM(sales)" in cleaned

    # Validate against allowed table
    valid = validate_sql(raw_response_with_fence, allowed_tables=["data_table"])
    assert "SELECT" in valid


def test_scenario_12_sql_syntax_errors():
    """Scenario 12: SQL syntax errors are caught during AST parsing."""
    bad_syntax_queries = [
        "SELEC region FORM data_table",
        "SELECT FROM WHERE",
        "SELECT * FROM WHERE",
        "SELECT ) FROM data_table"
    ]
    for bad_sql in bad_syntax_queries:
        with pytest.raises(SQLValidationError):
            validate_sql(bad_sql, allowed_tables=["data_table"])


def test_scenario_13_unsafe_sql_statements():
    """Scenario 13: Unsafe modifying statements and stacked injections are strictly rejected."""
    unsafe_queries = [
        "DROP TABLE data_table;",
        "DELETE FROM data_table WHERE 1=1;",
        "UPDATE data_table SET sales = 0;",
        "INSERT INTO data_table (region) VALUES ('Fake');",
        "ALTER TABLE data_table ADD COLUMN hacked TEXT;",
        "PRAGMA table_info(data_table);",
        "SELECT * FROM data_table; DROP TABLE data_table;",  # Stacked injection
        "ATTACH DATABASE 'malicious.db' AS mal;",
        "CREATE TABLE secret AS SELECT * FROM data_table;"
    ]

    for q in unsafe_queries:
        with pytest.raises(SQLValidationError):
            validate_sql(q, allowed_tables=["data_table"])


def test_table_access_restriction():
    """Verifies queries attempting to read unauthorized tables are rejected."""
    query = "SELECT * FROM users WHERE admin = 1;"
    with pytest.raises(SQLValidationError, match="unauthorized or non-existent table"):
        validate_sql(query, allowed_tables=["data_table"])


def test_unanswerable_question_detection():
    """Verifies that an unanswerable marker from LLM raises UnanswerableQuestionError."""
    unanswerable_response = "-- UNANSWERABLE: The dataset does not contain weather or temperature information."
    with pytest.raises(UnanswerableQuestionError):
        clean_sql_output(unanswerable_response)
