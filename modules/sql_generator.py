"""
SQL Generator Module
Constructs schema-aware prompts, interacts with the LLM service to produce SQLite queries,
and dynamically generates dataset-relevant question suggestions.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from .llm_service import GroqLLMService


class SQLGeneratorError(Exception):
    """Custom exception raised during SQL query generation."""
    pass


class SQLGenerator:
    """
    Translates natural language questions into safe, deterministic SQLite queries
    bound strictly to the active dataset schema.
    """

    def __init__(self, llm_service: GroqLLMService):
        self.llm_service = llm_service

    def build_system_prompt(self, table_name: str, schema_ddl: str, schema_metadata: Dict[str, Any]) -> str:
        """
        Builds a comprehensive system prompt enforcing SQLite dialect constraints,
        exact column adherence, and sample value awareness.
        """
        column_details: List[str] = []
        for col, info in schema_metadata.get("columns", {}).items():
            samples = ", ".join([f"'{s}'" for s in info.get("sample_values", [])])
            column_details.append(
                f"- Column \"{col}\" ({info.get('sqlite_type', 'TEXT')}), "
                f"non-null: {info.get('non_null_count', 0)}, sample values: [{samples}]"
            )

        col_context_str = "\n".join(column_details)

        prompt = f"""You are an expert SQL Data Analyst specializing in SQLite databases.
Your job is to convert the user's natural language question into a valid, safe, and accurate SQLite query.

### DATABASE SCHEMA:
Target Table Name: "{table_name}"

DDL:
{schema_ddl}

Column Schema & Sample Values:
{col_context_str}

### STRICT RULES & INSTRUCTIONS:
1. ONLY produce a single SQLite-compatible SELECT query.
2. You MUST use ONLY the table "{table_name}" and the exact columns listed above.
3. NEVER invent, hallucinate, or assume columns or tables that are not in the schema.
4. If column names contain special characters or spaces, wrap them in double quotes (e.g., "column_name").
5. Case Sensitivity: For string filtering (WHERE clauses), observe the sample values provided above. If uncertain of casing, use SQLite's LOWER() or COLLATE NOCASE.
6. For date operations, use SQLite date functions such as `strftime('%Y-%m', "date_col")` or `DATE("date_col")`.
7. NEVER output destructive or modifying SQL (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, PRAGMA, TRUNCATE).
8. If the user's question CANNOT be answered using the provided dataset schema, respond with:
   -- UNANSWERABLE: The question cannot be answered because required fields are not present in the dataset.
9. Output RAW SQL ONLY. Do NOT wrap your query in markdown fences like ```sql or ```. Do NOT write explanations or conversational greeting.
"""
        return prompt

    def generate_sql(
        self,
        question: str,
        table_name: str,
        schema_ddl: str,
        schema_metadata: Dict[str, Any]
    ) -> str:
        """
        Generates an SQLite query for the given user question.
        """
        if not question or not question.strip():
            raise SQLGeneratorError("Question cannot be empty.")

        system_prompt = self.build_system_prompt(table_name, schema_ddl, schema_metadata)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Question: {question.strip()}"}
        ]

        raw_sql = self.llm_service.generate_chat_completion(messages, temperature=0.0)
        return raw_sql

    @staticmethod
    def generate_suggested_questions(schema_metadata: Dict[str, Any], max_suggestions: int = 5) -> List[str]:
        """
        Dynamically derives relevant, dataset-specific sample questions based on
        actual detected columns, data types, and distributions.
        """
        columns = schema_metadata.get("columns", {})
        if not columns:
            return ["How many total records are present in the dataset?"]

        suggestions: List[str] = [
            "How many total records are in the dataset?"
        ]

        # Categorize columns
        cat_cols = []
        num_cols = []
        date_cols = []

        for col, info in columns.items():
            col_type = info.get("sqlite_type", "TEXT").upper()
            if col_type in ("INTEGER", "REAL"):
                num_cols.append(col)
            elif col_type == "TIMESTAMP" or "date" in col.lower() or "time" in col.lower():
                date_cols.append(col)
            else:
                cat_cols.append(col)

        # 1. Categorical + Numeric Aggregation
        if cat_cols and num_cols:
            c = cat_cols[0]
            n = num_cols[0]
            suggestions.append(f"What is the total {n} by {c}?")
            suggestions.append(f"Show the top 5 {c} by {n}.")

        # 2. Date Trends
        if date_cols and num_cols:
            d = date_cols[0]
            n = num_cols[0]
            suggestions.append(f"Show the monthly trend of {n} over time.")

        # 3. Numeric Summary
        if num_cols:
            n = num_cols[-1]
            suggestions.append(f"What is the average {n} across all records?")

        # 4. Multi-category breakdown
        if len(cat_cols) >= 2:
            suggestions.append(f"What is the distribution of records across {cat_cols[0]} and {cat_cols[1]}?")

        return suggestions[:max_suggestions]
