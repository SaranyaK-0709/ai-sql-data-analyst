"""
Tests for Visualizer Module
Covers Scenarios:
17. Chart generation for suitable results
18. Queries that cannot be visualized
"""

import pandas as pd
import pytest
from modules.visualizer import ChartSelector


def test_scenario_17_chart_generation_suitable_results():
    """Scenario 17: Suitable aggregation results recommend appropriate chart types."""
    # 1. Categorical + Numeric -> Bar / Pie
    df_cat_num = pd.DataFrame({
        "region": ["North", "South", "East", "West"],
        "total_sales": [12000.0, 8500.0, 9400.0, 11200.0]
    })
    rec_type, x_col, y_col, _ = ChartSelector.recommend_chart_type(df_cat_num)
    assert rec_type in ["pie", "bar"]
    assert x_col == "region"
    assert y_col == "total_sales"

    fig = ChartSelector.render_figure(df_cat_num, rec_type, x_col, y_col)
    assert fig is not None

    # 2. Time series (Date + Numeric) -> Line
    df_timeseries = pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=5),
        "revenue": [100, 150, 130, 190, 220]
    })
    rec_type, x_col, y_col, _ = ChartSelector.recommend_chart_type(df_timeseries)
    assert rec_type == "line"
    assert x_col == "order_date"
    assert y_col == "revenue"

    fig = ChartSelector.render_figure(df_timeseries, rec_type, x_col, y_col)
    assert fig is not None

    # 3. Single Scalar Value -> Metric
    df_scalar = pd.DataFrame({"total_count": [1450]})
    rec_type, x_col, y_col, _ = ChartSelector.recommend_chart_type(df_scalar)
    assert rec_type == "metric"


def test_scenario_18_queries_cannot_be_visualized():
    """Scenario 18: Queries returning raw text records or empty results default safely to table view."""
    # Text-heavy multi-column non-aggregated result
    df_unvisualizable = pd.DataFrame({
        "user_id": ["U1", "U2", "U3"],
        "bio": ["Engineer in SF", "Designer in NY", "Manager in TX"],
        "status": ["Active", "Pending", "Inactive"]
    })
    rec_type, x_col, y_col, rationale = ChartSelector.recommend_chart_type(df_unvisualizable)
    assert rec_type == "table"
    assert "table" in rationale.lower()

    # Empty DataFrame
    df_empty = pd.DataFrame()
    rec_type, _, _, _ = ChartSelector.recommend_chart_type(df_empty)
    assert rec_type == "table"


def test_advanced_chart_types():
    """Verify all advanced chart types render valid figures."""
    df = pd.DataFrame({
        "product": ["Widget A", "Widget B", "Widget C", "Widget D"],
        "sales": [1200.0, 3400.0, 2100.0, 4500.0],
        "quantity": [10, 25, 18, 40]
    })

    advanced_types = [
        "horizontal_bar",
        "histogram",
        "box",
        "treemap",
        "funnel",
        "density_heatmap",
        "area",
        "scatter"
    ]

    for chart_type in advanced_types:
        fig = ChartSelector.render_figure(df, chart_type, x_col="product", y_col="sales")
        assert fig is not None, f"Failed to render advanced chart type: {chart_type}"
