"""
Tests for Query Executor Module
Covers Scenarios:
14. Empty query results
19. Large result sets
Security: Authorizer enforcement & query timeout watchdog
"""

import sqlite3
import pytest
import pandas as pd
from modules.query_executor import QueryExecutor, QueryExecutionError, QueryTimeoutError


@pytest.fixture
def test_db():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE data_table (id INTEGER, region TEXT, sales REAL);")
    cursor.executemany(
        "INSERT INTO data_table VALUES (?, ?, ?);",
        [
            (1, "North", 150.0),
            (2, "South", 250.0),
            (3, "East", 350.0),
            (4, "West", 450.0)
        ]
    )
    conn.commit()
    yield conn
    conn.close()


def test_scenario_14_empty_query_results(test_db):
    """Scenario 14: Empty query results return a valid DataFrame with 0 rows."""
    executor = QueryExecutor(connection=test_db, timeout_seconds=5, max_rows=100)
    query = "SELECT * FROM data_table WHERE sales > 99999;"
    df, latency, count, truncated = executor.execute_query(query)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert count == 0
    assert not truncated
    assert list(df.columns) == ["id", "region", "sales"]


def test_scenario_19_large_result_sets(test_db):
    """Scenario 19: Large result sets are safely truncated to max_rows."""
    # Set max_rows = 2
    executor = QueryExecutor(connection=test_db, timeout_seconds=5, max_rows=2)
    query = "SELECT * FROM data_table;"
    df, latency, count, truncated = executor.execute_query(query)

    assert len(df) == 2
    assert count == 2
    assert truncated is True


def test_authorizer_defense_blocks_modification(test_db):
    """Verifies that engine-level authorizer blocks any modifying queries."""
    executor = QueryExecutor(connection=test_db, timeout_seconds=5, max_rows=100)
    unsafe_query = "DELETE FROM data_table WHERE id = 1;"

    with pytest.raises(QueryExecutionError, match="Security Violation|not authorized"):
        executor.execute_query(unsafe_query)

    # Ensure data was NOT modified
    cursor = test_db.cursor()
    cursor.execute("SELECT COUNT(*) FROM data_table;")
    assert cursor.fetchone()[0] == 4


def test_query_timeout_watchdog(test_db):
    """Verifies that a runaway query is aborted by the progress handler timeout."""
    # Set immediate timeout (0 seconds) with a cross join on data_table
    cursor = test_db.cursor()
    cursor.execute("CREATE TABLE big_t (x INT);")
    cursor.executemany("INSERT INTO big_t VALUES (?);", [(i,) for i in range(1000)])
    test_db.commit()

    executor = QueryExecutor(connection=test_db, timeout_seconds=0, max_rows=100000)
    query = "SELECT count(*) FROM big_t CROSS JOIN big_t;"
    with pytest.raises(QueryTimeoutError):
        executor.execute_query(query)
