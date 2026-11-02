"""
Tests for Insights Module
Verifies grounded statistical and AI narrative insight generation.
"""

from unittest.mock import MagicMock
import pandas as pd
from modules.insights import InsightsGenerator
from modules.llm_service import GroqLLMService


def test_statistical_insights_aggregation():
    """Verifies deterministic math facts (highest performer, lowest performer, share)."""
    df = pd.DataFrame({
        "region": ["North", "South", "East"],
        "sales": [5000.0, 3000.0, 2000.0]
    })
    facts = InsightsGenerator.generate_statistical_insights(df)

    assert any("North" in f and "Highest Performer" in f for f in facts)
    assert any("East" in f and "Lowest Performer" in f for f in facts)
    assert any("Market Share" in f for f in facts)


def test_scalar_kpi_insights():
    """Verifies insights for single scalar aggregate (e.g. COUNT or SUM)."""
    df = pd.DataFrame({"total_revenue": [125000.50]})
    facts = InsightsGenerator.generate_statistical_insights(df)
    assert any("total_revenue" in f and "125,000.50" in f for f in facts)


def test_grounded_ai_narrative_mock():
    """Verifies that AI narrative calls LLM with grounded prompt."""
    mock_llm = MagicMock(spec=GroqLLMService)
    mock_llm.generate_chat_completion.return_value = "- North region led overall sales with $5,000.\n- East recorded lowest sales at $2,000."

    df = pd.DataFrame({
        "region": ["North", "East"],
        "sales": [5000.0, 2000.0]
    })

    generator = InsightsGenerator(llm_service=mock_llm)
    narrative = generator.generate_ai_narrative("Sales by region", "SELECT region, sales FROM data_table", df)

    assert "North region" in narrative
    assert mock_llm.generate_chat_completion.called
