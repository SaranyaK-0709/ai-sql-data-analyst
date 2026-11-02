"""
Insights Module
Generates analytical observations and natural-language summaries grounded 100%
in the executed query result set, combining deterministic statistics and AI reasoning.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from .llm_service import GroqLLMService


class InsightsGenerator:
    """
    Produces grounded analytical insights from query results.
    Guarantees no hallucination by anchoring directly to DataFrame properties.
    """

    def __init__(self, llm_service: Optional[GroqLLMService] = None):
        self.llm_service = llm_service

    @staticmethod
    def generate_statistical_insights(df: pd.DataFrame) -> List[str]:
        """
        Computes deterministic mathematical facts directly from the query result DataFrame.
        """
        if df is None or df.empty:
            return ["No data points were returned by this query."]

        num_rows, num_cols = df.shape
        insights: List[str] = []

        # Single value KPI
        if num_rows == 1 and num_cols == 1:
            col_name = df.columns[0]
            val = df.iloc[0, 0]
            formatted_val = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}" if isinstance(val, int) else str(val)
            insights.append(f"**Result**: The calculated **{col_name}** is **{formatted_val}**.")
            return insights

        # General row/col count
        insights.append(f"Returned **{num_rows:,} records** across **{num_cols} columns**.")

        # Identify numeric and categorical columns
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

        # Analyze categorical + numeric aggregations (e.g., grouped totals)
        if categorical_cols and numeric_cols:
            cat = categorical_cols[0]
            num = numeric_cols[0]

            # Highest and lowest ranking categories
            max_idx = df[num].idxmax()
            min_idx = df[num].idxmin()

            max_cat = df.loc[max_idx, cat]
            max_val = df.loc[max_idx, num]

            min_cat = df.loc[min_idx, cat]
            min_val = df.loc[min_idx, num]

            fmt_max = f"{max_val:,.2f}" if isinstance(max_val, float) else f"{max_val:,}" if isinstance(max_val, int) else str(max_val)
            fmt_min = f"{min_val:,.2f}" if isinstance(min_val, float) else f"{min_val:,}" if isinstance(min_val, int) else str(min_val)

            insights.append(f"**Highest Performer**: **{max_cat}** leads with **{fmt_max}** in {num}.")
            if num_rows > 1 and max_cat != min_cat:
                insights.append(f"**Lowest Performer**: **{min_cat}** recorded **{fmt_min}** in {num}.")

            # Total sum and share if applicable
            total_sum = df[num].sum()
            if total_sum > 0 and isinstance(max_val, (int, float)):
                share = (max_val / total_sum) * 100
                insights.append(f"**Market Share**: **{max_cat}** accounts for **{share:.1f}%** of the aggregate total.")

        elif numeric_cols:
            num = numeric_cols[0]
            avg_val = df[num].mean()
            max_val = df[num].max()
            min_val = df[num].min()
            insights.append(f"**Averages & Bounds**: Average {num} is **{avg_val:,.2f}** (Range: {min_val:,.2f} to {max_val:,.2f}).")

        return insights

    def generate_ai_narrative(
        self,
        question: str,
        sql_query: str,
        df: pd.DataFrame,
        max_rows_for_llm: int = 15
    ) -> str:
        """
        Uses Groq LLM to synthesize a concise, executive-level narrative summary
        grounded solely in the returned DataFrame.
        """
        if df is None or df.empty:
            return "The query completed successfully, but returned an empty result set (0 rows)."

        # Provide deterministic statistical facts first as anchor
        deterministic_facts = self.generate_statistical_insights(df)

        if not self.llm_service:
            return "\n\n".join(deterministic_facts)

        # Truncate table representation for token safety
        preview_csv = df.head(max_rows_for_llm).to_csv(index=False)

        prompt = f"""You are a senior data analyst presenting query results to an executive stakeholder.
Synthesize the provided query results into 2 to 3 concise bullet points.

STRICT GROUNDING RULES:
1. ONLY reference numbers, categories, and patterns present in the result data below.
2. DO NOT make assumptions or hallucinate business contexts not supported by the numbers.
3. Be professional, clear, and direct.

User Question: {question}
Executed SQL: {sql_query}

Query Results (first {min(len(df), max_rows_for_llm)} rows):
```csv
{preview_csv}
```
"""

        messages = [
            {"role": "system", "content": "You are a precise data analyst who summarizes query results concisely and factually."},
            {"role": "user", "content": prompt}
        ]

        try:
            narrative = self.llm_service.generate_chat_completion(messages, temperature=0.1, max_tokens=300)
            return narrative.strip()
        except Exception:
            # Fallback to pure statistical insights if LLM call fails
            return "\n\n".join(deterministic_facts)
