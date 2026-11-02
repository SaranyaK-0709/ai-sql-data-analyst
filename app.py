"""
AI SQL Data Analyst Agent (CSV -> SQL -> Insights)
A professional, streamlined application that converts natural language questions
into verified SQLite queries, interactive Plotly visualizations, and grounded insights.
"""

import os
import time
from typing import Optional, List, Dict, Any
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import importlib
import modules.visualizer
importlib.reload(modules.visualizer)
from modules.data_loader import load_csv, get_schema_metadata, DataLoaderError
from modules.data_cleaner import DataCleaner, DataCleanerError
from modules.database import SQLiteManager, DatabaseError
from modules.llm_service import GroqLLMService, LLMServiceError
from modules.sql_generator import SQLGenerator, SQLGeneratorError
from modules.sql_validator import validate_sql, SQLValidationError, UnanswerableQuestionError
from modules.query_executor import QueryExecutor, QueryExecutionError, QueryTimeoutError
from modules.visualizer import ChartSelector
from modules.insights import InsightsGenerator

# Load environment variables
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI SQL Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme-Safe Styling
st.markdown("""
<style>
    /* Clean spacing adjustments without breaking dark/light mode */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Compact, sleek metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        line-height: 1.2 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        opacity: 0.85 !important;
    }
    [data-testid="stMetric"] {
        padding: 8px 14px !important;
        border-radius: 8px !important;
        background: rgba(125, 125, 125, 0.08) !important;
        border: 1px solid rgba(125, 125, 125, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "db_manager" not in st.session_state:
    st.session_state.db_manager = SQLiteManager()
if "cleaner" not in st.session_state:
    st.session_state.cleaner = None
if "schema_metadata" not in st.session_state:
    st.session_state.schema_metadata = {}
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def sync_dataframe(df: pd.DataFrame):
    """Loads dataframe into SQLite and refreshes schema metadata."""
    st.session_state.df = df
    st.session_state.db_manager.load_dataframe(df, table_name="data_table")
    meta = get_schema_metadata(df)
    st.session_state.schema_metadata = meta
    st.session_state.suggested_questions = SQLGenerator.generate_suggested_questions(meta)


def run_query(question: str):
    """Executes full pipeline: Question -> Groq SQL -> Validation -> SQLite Execution -> Visuals -> Insights."""
    if not question or not question.strip():
        st.warning("Please enter a question.")
        return

    try:
        with st.spinner("Generating SQL query and executing..."):
            # 1. Initialize LLM Service
            llm_service = GroqLLMService()
            sql_gen = SQLGenerator(llm_service=llm_service)

            # 2. Build schema DDL and generate SQL
            ddl = st.session_state.db_manager.get_table_schema_ddl("data_table")
            meta = st.session_state.schema_metadata
            raw_sql = sql_gen.generate_sql(question, "data_table", ddl, meta)

            # 3. Clean and validate SQL (Read-only single statement enforcement)
            validated_sql = validate_sql(raw_sql, allowed_tables=["data_table"])

            # 4. Execute validated SQL with timeout and row limit safeguards
            executor = QueryExecutor(
                connection=st.session_state.db_manager.get_connection(),
                timeout_seconds=10,
                max_rows=5000
            )
            result_df, latency_ms, row_count, truncated = executor.execute_query(validated_sql)

            # 5. Generate Grounded AI Insights
            insights_gen = InsightsGenerator(llm_service=llm_service)
            narrative = insights_gen.generate_ai_narrative(question, validated_sql, result_df)

            # 6. Store result
            result_item = {
                "question": question,
                "sql": validated_sql,
                "df": result_df,
                "narrative": narrative,
                "latency_ms": latency_ms,
                "row_count": row_count,
                "timestamp": time.strftime("%H:%M:%S")
            }
            st.session_state.last_result = result_item
            st.session_state.query_history.insert(0, result_item)

    except UnanswerableQuestionError as uqe:
        st.warning(f"⚠️ {uqe}")
    except (SQLValidationError, SQLGeneratorError) as ve:
        st.error(f"❌ SQL Validation Guard: {ve}")
    except (QueryExecutionError, QueryTimeoutError) as qe:
        st.error(f"❌ Execution Error: {qe}")
    except LLMServiceError as le:
        st.error(f"❌ Groq AI Service Error: {le}")
    except Exception as ex:
        st.error(f"❌ Error: {ex}")


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 📁 Dataset Management")

    data_source = st.radio(
        "Choose Dataset:",
        ["Sales Data (Default Sample)", "Customer Churn (Sample)", "Upload Custom CSV"],
        index=0
    )

    if data_source == "Sales Data (Default Sample)":
        sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sales_data.csv")
        if os.path.exists(sample_path) and (st.session_state.get("_loaded_source") != "sales"):
            df, mapping = load_csv(sample_path)
            st.session_state.original_df = df.copy()
            st.session_state.cleaner = DataCleaner(df)
            sync_dataframe(df)
            st.session_state._loaded_source = "sales"

    elif data_source == "Customer Churn (Sample)":
        sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "customer_churn.csv")
        if os.path.exists(sample_path) and (st.session_state.get("_loaded_source") != "churn"):
            df, mapping = load_csv(sample_path)
            st.session_state.original_df = df.copy()
            st.session_state.cleaner = DataCleaner(df)
            sync_dataframe(df)
            st.session_state._loaded_source = "churn"

    elif data_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            if st.session_state.df is None or st.session_state.get("_uploaded_name") != uploaded_file.name:
                try:
                    df, mapping = load_csv(uploaded_file)
                    st.session_state.original_df = df.copy()
                    st.session_state.cleaner = DataCleaner(df)
                    sync_dataframe(df)
                    st.session_state._uploaded_name = uploaded_file.name
                    st.session_state._loaded_source = "custom"
                    st.success(f"Loaded '{uploaded_file.name}'")
                except DataLoaderError as dle:
                    st.error(f"Error loading CSV: {dle}")

    st.markdown("---")

    # Dataset Summary Metrics
    if st.session_state.df is not None:
        st.markdown("### 📊 Dataset Details")
        st.markdown(f"**Total Rows:** `{len(st.session_state.df):,}`")
        st.markdown(f"**Total Columns:** `{len(st.session_state.df.columns)}`")
        st.markdown(f"**Table Name:** `data_table`")

        if st.button("🔄 Reset / Clear Session", use_container_width=True):
            st.session_state.df = None
            st.session_state.original_df = None
            st.session_state.last_result = None
            st.session_state.query_history = []
            st.session_state._loaded_source = None
            st.session_state.db_manager.close()
            st.session_state.db_manager = SQLiteManager()
            st.rerun()


# --- MAIN CONTENT AREA ---
st.title("📊 AI SQL Data Analyst Agent")
st.markdown("Ask questions about your dataset in plain English. The AI generates and executes SQL to deliver instant answers, visualizations, and insights.")

if st.session_state.df is None:
    st.info("👋 To get started, select or upload a dataset using the left sidebar.")
    st.stop()


# 1. Dataset Preview (Expandable)
with st.expander("👁️ Dataset Preview & Columns", expanded=False):
    st.dataframe(st.session_state.df.head(15), use_container_width=True)
    cols_display = ", ".join([f"`{c}`" for c in st.session_state.df.columns])
    st.markdown(f"**Available Columns:** {cols_display}")


# 2. Suggested Questions
if st.session_state.suggested_questions:
    st.markdown("**Suggested Questions:**")
    cols = st.columns(len(st.session_state.suggested_questions[:4]))
    for i, q in enumerate(st.session_state.suggested_questions[:4]):
        if cols[i].button(q, key=f"sug_{i}", use_container_width=True):
            st.session_state.current_question = q
            run_query(q)
            st.rerun()


# 3. Question Input Form (Using st.form for 100% reliable submissions)
with st.form("query_form", clear_on_submit=False):
    question_text = st.text_input(
        "Enter your question in natural English:",
        value=st.session_state.current_question,
        placeholder="e.g. What is the total sales by region?",
    )
    submit_button = st.form_submit_button("🚀 Run Query", type="primary")

if submit_button:
    st.session_state.current_question = question_text
    run_query(question_text)


# 4. Display Results
if st.session_state.last_result is not None:
    res = st.session_state.last_result
    result_df = res["df"]

    st.markdown("---")
    st.markdown(f"### 📊 Results for: *\"{res['question']}\"*")

    # Metrics Row
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("⏱️ Execution Time", f"{res['latency_ms']} ms")
    m_col2.metric("📋 Rows Returned", f"{res['row_count']:,}")
    m_col3.metric("🕒 Timestamp", res["timestamp"])

    # Generated SQL
    with st.expander("🛠️ View Generated SQL Query", expanded=True):
        st.code(res["sql"], language="sql")

    # Interactive Visualization
    rec_type, rec_x, rec_y, rationale = ChartSelector.recommend_chart_type(result_df)

    if rec_type != "table" and not result_df.empty:
        st.markdown("#### 📈 Interactive Visualization")
        st.caption(f"💡 **AI Recommendation:** {rationale}")

        v_col1, v_col2, v_col3 = st.columns([1, 1, 1])
        with v_col1:
            chart_types = [
                "recommended",
                "bar",
                "horizontal_bar",
                "line",
                "pie",
                "area",
                "scatter",
                "histogram",
                "box",
                "treemap",
                "funnel",
                "density_heatmap",
                "table"
            ]
            chart_labels = {
                "recommended": f"Auto-Detect ({rec_type.replace('_', ' ').title()})",
                "bar": "Bar Chart (Vertical)",
                "horizontal_bar": "Bar Chart (Horizontal)",
                "line": "Line Chart",
                "pie": "Pie / Donut Chart",
                "area": "Area Chart",
                "scatter": "Scatter Plot",
                "histogram": "Histogram (Distribution)",
                "box": "Box Plot (IQR & Outliers)",
                "treemap": "Treemap (Proportions)",
                "funnel": "Funnel Chart",
                "density_heatmap": "Density Heatmap",
                "table": "Table View"
            }
            selected_chart = st.selectbox(
                "Chart Type:",
                chart_types,
                format_func=lambda x: chart_labels.get(x, x.title())
            )

        active_type = rec_type if selected_chart == "recommended" else selected_chart
        all_cols = list(result_df.columns)

        with v_col2:
            x_axis = st.selectbox(
                "X-Axis:",
                all_cols,
                index=all_cols.index(rec_x) if rec_x in all_cols else 0
            ) if active_type not in ("metric", "table") and all_cols else None

        with v_col3:
            numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(result_df[c])]
            y_options = numeric_cols if numeric_cols else all_cols
            y_axis = st.selectbox(
                "Y-Axis:",
                y_options,
                index=y_options.index(rec_y) if rec_y in y_options else 0
            ) if active_type not in ("metric", "table") and y_options else None

        if active_type != "table":
            fig = ChartSelector.render_figure(
                result_df,
                chart_type=active_type,
                x_col=x_axis,
                y_col=y_axis,
                title=res["question"]
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{active_type}")
            else:
                st.warning("⚠️ This chart type could not be rendered with the currently selected columns. Try choosing a different X-Axis or Y-Axis column above.")

    # Results Table
    st.markdown("#### 📋 Query Results Table")
    st.dataframe(result_df, use_container_width=True)

    # Export Results as CSV
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_bytes,
        file_name="query_results.csv",
        mime="text/csv"
    )

    # Grounded AI Insights
    st.markdown("#### 💡 AI-Generated Insights")
    st.info(res["narrative"], icon="💡")


# 5. Query History
if len(st.session_state.query_history) > 1:
    st.markdown("---")
    with st.expander("📜 Recent Query History", expanded=False):
        for i, h in enumerate(st.session_state.query_history[1:6]):
            st.markdown(f"**Question:** {h['question']}")
            st.code(h["sql"], language="sql")
            st.caption(f"Time: {h['timestamp']} | Latency: {h['latency_ms']} ms | Rows: {h['row_count']}")
            st.divider()
