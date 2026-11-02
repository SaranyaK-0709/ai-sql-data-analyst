"""
Tests for Autonomous Data Analyst Agent Module
Verifies multi-turn conversational memory, autonomous auto-pilot discovery,
self-healing reflection loop, and executive dossier export.
"""

from unittest.mock import MagicMock
import sqlite3
import pandas as pd
import pytest

from modules.agent import AutonomousDataAnalystAgent
from modules.llm_service import GroqLLMService


@pytest.fixture
def agent_fixture():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE data_table (id INT, region TEXT, sales REAL, product TEXT);")
    conn.executemany(
        "INSERT INTO data_table VALUES (?, ?, ?, ?);",
        [
            (1, "North", 1000.0, "Laptop"),
            (2, "South", 2000.0, "Phone"),
            (3, "East", 3000.0, "Laptop"),
            (4, "West", 4000.0, "Tablet")
        ]
    )
    conn.commit()

    df = pd.read_sql_query("SELECT * FROM data_table", conn)

    mock_llm = MagicMock(spec=GroqLLMService)
    # Default valid response
    mock_llm.generate_chat_completion.return_value = "SELECT region, SUM(sales) FROM data_table GROUP BY region;"

    agent = AutonomousDataAnalystAgent(
        llm_service=mock_llm,
        connection=conn,
        table_name="data_table",
        schema_ddl="CREATE TABLE data_table (id INT, region TEXT, sales REAL, product TEXT);",
        schema_metadata={"columns": {"region": {"sqlite_type": "TEXT"}, "sales": {"sqlite_type": "REAL"}}}
    )

    yield agent, df, mock_llm
    conn.close()


def test_autonomous_autopilot_briefing(agent_fixture):
    """Verifies autonomous scanning extracts KPIs and top growth drivers without user prompt."""
    agent, df, _ = agent_fixture
    brief = agent.run_autopilot_briefing(df)

    assert "Total Rows" in brief["kpis"]
    assert "Aggregate Sales" in brief["kpis"]
    assert len(brief["highlights"]) >= 1
    assert any("West" in h and "Driver" in h for h in brief["highlights"])


def test_multi_turn_conversational_memory(agent_fixture):
    """Verifies that conversation history is preserved and injected into subsequent queries."""
    agent, _, mock_llm = agent_fixture

    # Turn 1
    res1 = agent.process_query("What is the sales by region?")
    assert res1.row_count == 4
    assert len(agent.conversation_history) == 1

    # Turn 2: Follow-up referencing Turn 1
    mock_llm.generate_chat_completion.return_value = "SELECT region, SUM(sales) FROM data_table GROUP BY region ORDER BY SUM(sales) DESC LIMIT 2;"
    res2 = agent.process_query("Now filter only to the top 2 regions")

    assert res2.row_count == 2
    assert len(agent.conversation_history) == 2

    # Verify recent context was passed to the LLM system prompt in Turn 2
    # call_args_list[-2] is the SQL generation call; call_args_list[-1] is the narrative call
    sql_call_messages = mock_llm.generate_chat_completion.call_args_list[-2][0][0]
    system_prompt = sql_call_messages[0]["content"]
    assert "CONVERSATION CONTEXT" in system_prompt
    assert "What is the sales by region?" in system_prompt


def test_self_healing_reflection_loop(agent_fixture):
    """Verifies that when initial SQL fails, the agent autonomously reflects, fixes it, and re-executes."""
    agent, _, mock_llm = agent_fixture

    # Simulate: Attempt 1 produces bad SQL with invalid column, Attempt 2 fixes it
    mock_llm.generate_chat_completion.side_effect = [
        "SELECT non_existent_col FROM data_table;",  # Fails execution
        "SELECT region, SUM(sales) FROM data_table GROUP BY region;"  # Self-healed fix
    ]

    result = agent.process_query("Show me regional sales")
    assert result.self_healed is True
    assert result.row_count == 4
    assert any("self-healing" in step.lower() for step in result.reasoning_steps)


def test_executive_html_dossier_export(agent_fixture):
    """Verifies that session findings are compiled into a publication-ready HTML dossier."""
    agent, _, _ = agent_fixture
    agent.process_query("Sales by region")

    html_dossier = agent.export_executive_html_report(dataset_name="Retail Sales")
    assert "<!DOCTYPE html>" in html_dossier
    assert "Executive Analytics Dossier" in html_dossier
    assert "Sales by region" in html_dossier
    assert "styled-table" in html_dossier
