"""
Visualizer Module
Intelligently inspects SQL query results and selects the optimal interactive Plotly chart.
Supports Bar, Line, Donut, Scatter, Area charts, and Metric KPIs with customization overrides.
"""

from typing import Optional, Dict, Any, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


CHART_THEME = "plotly_white"


class ChartSelector:
    """
    Analyzes DataFrame geometry, column types, and distributions
    to recommend and render appropriate interactive Plotly visualizations.
    """

    @staticmethod
    def recommend_chart_type(df: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str], str]:
        """
        Heuristically determines the most suitable visualization for a query result.

        Returns:
            Tuple of:
            - recommended_type ('metric', 'bar', 'line', 'pie', 'scatter', 'table')
            - x_column
            - y_column
            - rationale explanation
        """
        if df is None or df.empty:
            return "table", None, None, "The query result set is empty."

        num_rows, num_cols = df.shape

        # Case 1: Single scalar value (1 row, 1 column) -> KPI metric
        if num_rows == 1 and num_cols == 1:
            col = df.columns[0]
            val = df.iloc[0, 0]
            return "metric", col, col, f"Single scalar value detected for '{col}'."

        # Categorize column types
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        datetime_cols = [
            c for c in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[c])
            or "date" in str(c).lower()
            or "month" in str(c).lower()
            or "year" in str(c).lower()
        ]
        categorical_cols = [
            c for c in df.columns
            if c not in numeric_cols and c not in datetime_cols
        ]

        # Case 2: Time-based trend (Date + Numeric) -> Line Chart
        if datetime_cols and numeric_cols:
            x_col = datetime_cols[0]
            y_col = numeric_cols[0]
            return "line", x_col, y_col, f"Time series trend of '{y_col}' over '{x_col}'."

        # Case 3: Categorical comparison (Categorical + Numeric)
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            unique_cats = df[cat_col].nunique()

            # Small proportion of categories -> Donut / Pie chart
            if 2 <= unique_cats <= 6 and num_rows <= 8:
                return "pie", cat_col, num_col, f"Proportional composition of '{num_col}' across '{cat_col}'."

            # Moderate to large number of categories -> Bar chart
            return "bar", cat_col, num_col, f"Categorical comparison of '{num_col}' grouped by '{cat_col}'."

        # Case 4: Correlation / Relationship (2 or more Numeric columns) -> Scatter
        if len(numeric_cols) >= 2 and not categorical_cols:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            return "scatter", x_col, y_col, f"Correlation scatter plot between '{x_col}' and '{y_col}'."

        # Case 5: Default to Table View for raw records or text-heavy outputs
        return "table", None, None, "Multi-dimensional or non-aggregated tabular records are best displayed as a table."

    @classmethod
    def render_figure(
        cls,
        df: pd.DataFrame,
        chart_type: str,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        title: Optional[str] = None
    ) -> Optional[go.Figure]:
        """
        Renders a Plotly Figure based on chosen chart type and column bindings.
        """
        if df is None or df.empty:
            return None

        clean_title = title or f"{chart_type.title()} Chart Analysis"

        try:
            chart_key = chart_type.lower()

            if chart_key == "recommended":
                rec_t, rx, ry, _ = cls.recommend_chart_type(df)
                chart_key = rec_t.lower()
                x_col = x_col or rx
                y_col = y_col or ry

            if chart_key == "bar":
                if not x_col or not y_col:
                    return None
                fig = px.bar(
                    df,
                    x=x_col,
                    y=y_col,
                    title=clean_title,
                    text_auto=True,
                    template=CHART_THEME,
                    color=x_col if df[x_col].nunique() <= 15 else None,
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig.update_layout(
                    showlegend=False,
                    xaxis_tickangle=-30,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

            elif chart_key in ("horizontal_bar", "bar_horizontal"):
                if not x_col or not y_col:
                    return None
                is_x_num = pd.api.types.is_numeric_dtype(df[x_col])
                is_y_num = pd.api.types.is_numeric_dtype(df[y_col])
                bar_x = x_col if is_x_num and not is_y_num else y_col
                bar_y = y_col if is_x_num and not is_y_num else x_col
                fig = px.bar(
                    df,
                    x=bar_x,
                    y=bar_y,
                    orientation="h",
                    title=clean_title,
                    text_auto=True,
                    template=CHART_THEME,
                    color=bar_y if df[bar_y].nunique() <= 15 else None,
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig.update_layout(
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

            elif chart_key == "line":
                if not x_col or not y_col:
                    return None
                fig = px.line(
                    df,
                    x=x_col,
                    y=y_col,
                    markers=True,
                    title=clean_title,
                    template=CHART_THEME,
                    color_discrete_sequence=["#3B82F6"]
                )
                fig.update_layout(
                    xaxis_tickangle=-30,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

            elif chart_key == "pie":
                if not x_col or not y_col:
                    return None
                is_x_num = pd.api.types.is_numeric_dtype(df[x_col])
                is_y_num = pd.api.types.is_numeric_dtype(df[y_col])
                val_col = x_col if is_x_num and not is_y_num else y_col
                name_col = y_col if is_x_num and not is_y_num else x_col
                fig = px.pie(
                    df,
                    names=name_col,
                    values=val_col,
                    hole=0.4,
                    title=clean_title,
                    template=CHART_THEME,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

            elif chart_key == "scatter":
                if not x_col or not y_col:
                    return None
                is_num = pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col])
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    trendline="ols" if (is_num and len(df) >= 3) else None,
                    title=clean_title,
                    template=CHART_THEME,
                    color_discrete_sequence=["#F59E0B"]
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

            elif chart_key == "area":
                if not x_col or not y_col:
                    return None
                fig = px.area(
                    df,
                    x=x_col,
                    y=y_col,
                    title=clean_title,
                    template=CHART_THEME,
                    color_discrete_sequence=["#10B981"]
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

            elif chart_key == "histogram":
                target_col = y_col if (y_col and pd.api.types.is_numeric_dtype(df[y_col])) else x_col
                if not target_col:
                    return None
                fig = px.histogram(
                    df,
                    x=target_col,
                    marginal="box",
                    title=clean_title,
                    template=CHART_THEME,
                    color_discrete_sequence=["#6366F1"]
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

            elif chart_key == "box":
                if not x_col or not y_col:
                    return None
                is_x_num = pd.api.types.is_numeric_dtype(df[x_col])
                is_y_num = pd.api.types.is_numeric_dtype(df[y_col])
                val_col = x_col if is_x_num and not is_y_num else y_col
                cat_col = y_col if is_x_num and not is_y_num else x_col
                fig = px.box(
                    df,
                    x=cat_col,
                    y=val_col,
                    points="all",
                    title=clean_title,
                    template=CHART_THEME,
                    color=cat_col if df[cat_col].nunique() <= 15 else None,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

            elif chart_key == "treemap":
                if not x_col or not y_col:
                    return None
                plot_df = df.copy()
                is_x_num = pd.api.types.is_numeric_dtype(df[x_col])
                is_y_num = pd.api.types.is_numeric_dtype(df[y_col])
                cat_col = y_col if is_x_num and not is_y_num else x_col
                num_col = x_col if is_x_num and not is_y_num else y_col
                plot_df[cat_col] = plot_df[cat_col].astype(str)
                fig = px.treemap(
                    plot_df,
                    path=[cat_col],
                    values=num_col if (is_x_num or is_y_num) else None,
                    title=clean_title,
                    template=CHART_THEME,
                    color=num_col if (is_x_num or is_y_num) else None,
                    color_continuous_scale="Blues"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

            elif chart_key == "funnel":
                if not x_col or not y_col:
                    return None
                is_x_num = pd.api.types.is_numeric_dtype(df[x_col])
                is_y_num = pd.api.types.is_numeric_dtype(df[y_col])
                bar_x = x_col if is_x_num and not is_y_num else y_col
                bar_y = y_col if is_x_num and not is_y_num else x_col
                fig = px.funnel(
                    df,
                    x=bar_x,
                    y=bar_y,
                    title=clean_title,
                    template=CHART_THEME,
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

            elif chart_key in ("density_heatmap", "heatmap"):
                if not x_col or not y_col:
                    return None
                fig = px.density_heatmap(
                    df,
                    x=x_col,
                    y=y_col,
                    title=clean_title,
                    template=CHART_THEME,
                    color_continuous_scale="Viridis",
                    text_auto=True
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

            elif chart_key == "metric":
                val = df.iloc[0, 0]
                label = df.columns[0]
                formatted_val = f"{val:,.2f}" if isinstance(val, (int, float)) else str(val)
                fig = go.Figure(go.Indicator(
                    mode="number",
                    value=float(val) if isinstance(val, (int, float)) else None,
                    title={"text": label.replace("_", " ").title(), "font": {"size": 20}},
                    number={"valueformat": ",.2f" if isinstance(val, float) else ",d"} if isinstance(val, (int, float)) else None
                ))
                if not isinstance(val, (int, float)):
                    fig.add_annotation(
                        text=formatted_val,
                        font={"size": 40, "color": "#1f77b4"},
                        showarrow=False
                    )
                fig.update_layout(
                    template=CHART_THEME,
                    height=250,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

            return None

        except Exception as exc:
            import logging
            logging.error("Visualizer rendering failed for chart '%s': %s", chart_type, exc)
            return None
