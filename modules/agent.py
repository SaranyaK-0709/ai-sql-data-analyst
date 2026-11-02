"""
Autonomous AI Data Analyst Agent Module
Implements multi-turn conversational memory, an agentic ReAct loop with self-healing SQL reflection,
automated dataset auto-pilot briefing, and scenario simulation.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import time
import pandas as pd
import sqlite3

from .llm_service import GroqLLMService, LLMServiceError
from .sql_generator import SQLGenerator
from .sql_validator import validate_sql, clean_sql_output, SQLValidationError, UnanswerableQuestionError
from .query_executor import QueryExecutor, QueryExecutionError
from .insights import InsightsGenerator
from .visualizer import ChartSelector


class AgentTurnResult:
    """Represents the complete output of an agent interaction cycle."""

    def __init__(
        self,
        question: str,
        sql_query: str,
        df: pd.DataFrame,
        narrative: str,
        execution_latency_ms: float,
        row_count: int,
        reasoning_steps: List[str],
        follow_up_suggestions: List[str],
        self_healed: bool = False,
        error: Optional[str] = None
    ):
        self.question = question
        self.sql_query = sql_query
        self.df = df
        self.narrative = narrative
        self.execution_latency_ms = execution_latency_ms
        self.row_count = row_count
        self.reasoning_steps = reasoning_steps
        self.follow_up_suggestions = follow_up_suggestions
        self.self_healed = self_healed
        self.error = error
        self.timestamp = time.strftime("%H:%M:%S")


class AutonomousDataAnalystAgent:
    """
    Stateful conversational AI agent that maintains conversational context,
    performs self-healing query reflection, and generates proactive analytical findings.
    """

    def __init__(
        self,
        llm_service: GroqLLMService,
        connection: sqlite3.Connection,
        table_name: str = "data_table",
        schema_ddl: str = "",
        schema_metadata: Optional[Dict[str, Any]] = None
    ):
        self.llm_service = llm_service
        self.conn = connection
        self.table_name = table_name
        self.schema_ddl = schema_ddl
        self.schema_metadata = schema_metadata or {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.executor = QueryExecutor(connection=connection, timeout_seconds=10, max_rows=5000)
        self.insights_gen = InsightsGenerator(llm_service=llm_service)

    def run_autopilot_briefing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Proactively scans the dataset upon ingestion and synthesizes an executive briefing:
        - Key KPI Metrics
        - Primary categorical drivers
        - Top value distributions
        """
        if df is None or df.empty:
            return {"kpis": {}, "highlights": ["Dataset is empty."]}

        kpis: Dict[str, Any] = {
            "Total Rows": f"{len(df):,}",
            "Total Columns": len(df.columns)
        }

        # Detect primary numeric and categorical columns
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and "date" not in c.lower()]

        highlights: List[str] = []

        if numeric_cols:
            # Filter out identifier columns like 'id', 'customer_id' from aggregate KPI metrics
            substantive_metrics = [
                c for c in numeric_cols
                if not (c.lower() == "id" or c.lower().endswith("_id") or c.lower().endswith("id"))
            ]
            primary_metric = substantive_metrics[0] if substantive_metrics else numeric_cols[0]
            total_sum = df[primary_metric].sum()
            avg_val = df[primary_metric].mean()
            kpis[f"Aggregate {primary_metric.replace('_', ' ').title()}"] = (
                f"{total_sum:,.2f}" if isinstance(total_sum, float) else f"{total_sum:,}"
            )
            kpis[f"Average {primary_metric.replace('_', ' ').title()}"] = (
                f"{avg_val:,.2f}" if isinstance(avg_val, float) else f"{avg_val:,}"
            )

            # Check for top driver if categorical column exists
            if cat_cols:
                primary_dim = cat_cols[0]
                grouped = df.groupby(primary_dim)[primary_metric].sum().reset_index()
                if not grouped.empty:
                    top_row = grouped.sort_values(by=primary_metric, ascending=False).iloc[0]
                    top_name = top_row[primary_dim]
                    top_val = top_row[primary_metric]
                    share = (top_val / total_sum * 100) if total_sum > 0 else 0
                    highlights.append(
                        f"**Key Growth Driver:** **{top_name}** represents **{share:.1f}%** "
                        f"of total {primary_metric} ({top_val:,.2f})."
                    )

            # Check for distribution range
            min_val = df[primary_metric].min()
            max_val = df[primary_metric].max()
            highlights.append(
                f"**Variance Spectrum:** Values for **{primary_metric}** range from {min_val:,.2f} to {max_val:,.2f}."
            )

        if not highlights:
            highlights.append(f"Autonomous discovery completed across {len(df.columns)} dimensions and {len(df):,} records.")

        return {
            "kpis": kpis,
            "highlights": highlights
        }

    def process_query(self, user_question: str) -> AgentTurnResult:
        """
        Executes a full Agent cycle with Multi-Turn Context and Self-Correction Reflection.
        """
        reasoning_steps: List[str] = []
        reasoning_steps.append(f"Analyzing prompt: '{user_question}' against '{self.table_name}' schema.")

        # Build Context from recent turns
        recent_context_str = ""
        if self.conversation_history:
            recent_turns = self.conversation_history[-3:]
            lines = ["### CONVERSATION CONTEXT (Recent Turns):"]
            for turn in recent_turns:
                lines.append(f"User: {turn['question']}")
                lines.append(f"Generated SQL: {turn['sql']}")
            recent_context_str = "\n".join(lines) + "\n\n"

        column_details = []
        for col, info in self.schema_metadata.get("columns", {}).items():
            samples = ", ".join([f"'{s}'" for s in info.get("sample_values", [])])
            column_details.append(
                f"- \"{col}\" ({info.get('sqlite_type', 'TEXT')}), samples: [{samples}]"
            )
        col_context_str = "\n".join(column_details)

        system_prompt = f"""You are an Autonomous Senior SQL Data Analyst Agent.
Your objective is to generate accurate, deterministic SQLite queries to answer user questions.

### DATABASE SPECIFICATION:
Table Name: "{self.table_name}"
DDL:
{self.schema_ddl}

Columns:
{col_context_str}

{recent_context_str}### INSTRUCTIONS:
1. Generate a single SQLite SELECT statement answering the user's question.
2. If the user refers to previous context (e.g. 'filter only the top 2', 'now group that by product'), adapt the previous query accordingly.
3. NEVER produce destructive or modifying queries (INSERT, UPDATE, DELETE, DROP, ALTER, PRAGMA).
4. Use double quotes for column and table names.
5. If the question cannot be answered from the schema, output:
   -- UNANSWERABLE: <brief reason>
6. Output raw SQL only without markdown code fences or conversational greetings.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]

        # Step 1: Draft initial SQL
        reasoning_steps.append("Generating candidate SQL query via Groq inference.")
        raw_sql = self.llm_service.generate_chat_completion(messages, temperature=0.0)

        # Step 2: Validate & Self-Heal loop (up to 2 correction attempts)
        current_sql = raw_sql
        validated_sql = None
        result_df = None
        latency_ms = 0.0
        row_count = 0
        truncated = False
        self_healed = False
        max_attempts = 2

        for attempt in range(max_attempts + 1):
            try:
                # AST Safety and Single Statement check
                validated_sql = validate_sql(current_sql, allowed_tables=[self.table_name])
                reasoning_steps.append(f"AST validation passed: Single read-only query verified.")

                # Execute with engine-level authorizer
                result_df, latency_ms, row_count, truncated = self.executor.execute_query(validated_sql)
                reasoning_steps.append(f"Query executed successfully ({row_count:,} records in {latency_ms}ms).")
                break

            except (SQLValidationError, QueryExecutionError) as err:
                if attempt < max_attempts:
                    reasoning_steps.append(f"⚠️ Execution encounter: {err}. Initiating autonomous reflection and self-healing...")
                    # Reflection prompt
                    heal_prompt = f"""The previous SQL query failed:
SQL: {current_sql}
Error: {str(err)}

Please inspect the error and provide a corrected, valid SQLite SELECT query.
Output RAW SQL ONLY without markdown formatting.
"""
                    heal_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": heal_prompt}
                    ]
                    current_sql = self.llm_service.generate_chat_completion(heal_messages, temperature=0.0)
                    self_healed = True
                else:
                    # If exhausted, raise
                    raise err

        # Step 3: Grounded Narrative Synthesis
        reasoning_steps.append("Synthesizing grounded business insights from executed result set.")
        narrative = self.insights_gen.generate_ai_narrative(
            question=user_question,
            sql_query=validated_sql,
            df=result_df
        )

        # Step 4: Autonomous Contextual Follow-up Suggestions
        follow_ups = self._generate_follow_up_recommendations(user_question, validated_sql, result_df)

        turn_result = AgentTurnResult(
            question=user_question,
            sql_query=validated_sql,
            df=result_df,
            narrative=narrative,
            execution_latency_ms=latency_ms,
            row_count=row_count,
            reasoning_steps=reasoning_steps,
            follow_up_suggestions=follow_ups,
            self_healed=self_healed
        )

        # Update Conversation Memory
        self.conversation_history.append({
            "question": user_question,
            "sql": validated_sql,
            "rows": row_count,
            "df": result_df,
            "narrative": narrative
        })

        return turn_result

    def _generate_follow_up_recommendations(
        self,
        question: str,
        sql: str,
        df: pd.DataFrame
    ) -> List[str]:
        """Dynamically formulates 3 logical next-step analytical drill-downs."""
        suggestions: List[str] = []
        if df is None or df.empty:
            return ["Show all records from data_table", "What are the columns in this dataset?"]

        cols = list(df.columns)
        cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

        # If grouping occurred, suggest sorting or filtering top
        if cat_cols and num_cols and len(df) > 1:
            top_cat = df.iloc[0][cat_cols[0]]
            suggestions.append(f"Drill down into records for '{top_cat}'")
            suggestions.append(f"Show only the top 3 {cat_cols[0]} by {num_cols[0]}")
            suggestions.append(f"What percentage share does each {cat_cols[0]} represent?")
        elif num_cols:
            suggestions.append(f"What is the average and median {num_cols[0]}?")
            suggestions.append(f"Show the top 5 records with highest {num_cols[0]}")
        else:
            suggestions.append("Count total records grouped by primary dimension")

        return suggestions[:3]

    def export_executive_html_report(self, dataset_name: str = "Active Dataset") -> str:
        """
        Compiles the entire session's conversation history, KPIs, SQL queries,
        and business findings into a standalone, publication-grade Executive Dossier (HTML).
        """
        session_turns = self.conversation_history
        if not session_turns:
            return "<html><body><h1>No queries recorded in session.</h1></body></html>"

        history_cards_html = []
        for i, turn in enumerate(session_turns):
            table_preview = turn["df"].head(10).to_html(classes="styled-table", index=False)
            history_cards_html.append(f"""
            <div class="query-card">
                <div class="query-header">
                    <span class="query-num">Query #{i+1}</span>
                    <span class="query-title">{turn['question']}</span>
                </div>
                <div class="sql-box">
                    <code>{turn['sql']}</code>
                </div>
                <div class="insights-box">
                    <h4>Executive Finding</h4>
                    <p>{turn['narrative'].replace(chr(10), '<br>')}</p>
                </div>
                <div class="table-container">
                    {table_preview}
                </div>
            </div>
            """)

        all_cards_str = "\n".join(history_cards_html)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Executive Analytics Dossier — {dataset_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #F8FAFC;
            color: #0F172A;
            margin: 0;
            padding: 40px;
        }}
        .header {{
            border-bottom: 2px solid #E2E8F0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .brand {{
            font-size: 24px;
            font-weight: 700;
            color: #2563EB;
        }}
        .meta {{
            font-size: 14px;
            color: #64748B;
            margin-top: 5px;
        }}
        .query-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 30px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }}
        .query-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }}
        .query-num {{
            background: #EFF6FF;
            color: #2563EB;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
        }}
        .query-title {{
            font-size: 18px;
            font-weight: 600;
        }}
        .sql-box {{
            background: #0F172A;
            color: #38BDF8;
            padding: 12px 16px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 13px;
            margin-bottom: 16px;
        }}
        .insights-box {{
            background: #F0FDF4;
            border-left: 4px solid #16A34A;
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 16px;
        }}
        .insights-box h4 {{
            margin: 0 0 6px 0;
            color: #166534;
            font-size: 14px;
        }}
        .styled-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .styled-table th, .styled-table td {{
            padding: 8px 12px;
            border: 1px solid #E2E8F0;
            text-align: left;
        }}
        .styled-table th {{
            background-color: #F1F5F9;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="brand">⚡ DataInsight AI — Executive Dossier</div>
        <div class="meta">Dataset: <strong>{dataset_name}</strong> | Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    {all_cards_str}
</body>
</html>
"""
        return html
