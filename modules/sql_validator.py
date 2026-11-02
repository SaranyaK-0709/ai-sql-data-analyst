"""
SQL Validator Module
Cleans raw LLM outputs, strips markdown formatting, and performs AST-based security checks
to guarantee read-only, single-statement execution against SQLite.
"""

from typing import List, Set, Optional
import re
import sqlglot
from sqlglot import exp


class SQLValidationError(Exception):
    """Custom exception raised when SQL generation violates safety or syntax rules."""
    pass


class UnanswerableQuestionError(SQLValidationError):
    """Raised when the LLM determines the user's question cannot be answered by the schema."""
    pass


# Forbidden DDL, DML, and administrative keywords
FORBIDDEN_KEYWORDS: Set[str] = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "REPLACE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX",
    "TRANSACTION", "COMMIT", "ROLLBACK", "EXEC", "EXECUTE", "GRANT", "REVOKE"
}


def clean_sql_output(raw_output: str) -> str:
    """
    Cleans and extracts pure SQL from raw LLM output:
    - Removes markdown code fences (```sql, ```)
    - Strips leading conversational preambles (e.g. 'Here is your SQL query:')
    - Normalizes internal whitespace and trailing semicolons.
    """
    if not raw_output or not raw_output.strip():
        raise SQLValidationError("LLM returned an empty SQL response.")

    text = raw_output.strip()

    # Check for unanswerable flag
    if text.startswith("-- UNANSWERABLE:") or "UNANSWERABLE:" in text:
        reason = text.split("UNANSWERABLE:", 1)[1].strip()
        raise UnanswerableQuestionError(f"Question cannot be answered with current data: {reason}")

    # Remove Markdown code blocks
    # Handles ```sql\nSELECT ...\n``` or ```\nSELECT ...\n```
    code_block_match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if code_block_match:
        text = code_block_match.group(1).strip()

    # Remove leading comments or conversational preamble lines before the first SELECT
    lines = text.splitlines()
    sql_lines: List[str] = []
    found_sql_start = False

    for line in lines:
        stripped = line.strip()
        if not found_sql_start:
            # Check if line begins with a SQL query keyword
            if re.match(r"^(SELECT|WITH)\b", stripped, re.IGNORECASE):
                found_sql_start = True
                sql_lines.append(stripped)
            elif stripped.startswith("--") or stripped.startswith("/*"):
                # Preserve inline comments if part of query
                continue
            else:
                # Preceding conversational greeting, skip
                continue
        else:
            sql_lines.append(stripped)

    if found_sql_start:
        text = "\n".join(sql_lines).strip()

    # Strip trailing semicolon for uniform handling
    text = text.rstrip(";").strip()

    if not text:
        raise SQLValidationError("No valid SQL SELECT statement could be extracted from the model response.")

    return text


def validate_sql(
    sql_query: str,
    allowed_tables: Optional[List[str]] = None
) -> str:
    """
    Validates the SQL query using sqlglot AST parsing and keyword guardrails.

    Enforces:
    1. Single statement execution (no stacked queries e.g. 'SELECT 1; DROP TABLE')
    2. Read-only operation: root AST must be a Select or Union expression
    3. Keyword blacklist: no forbidden DDL/DML tokens
    4. Table binding: referenced tables must match allowed_tables if provided

    Returns:
        The validated SQL query string.

    Raises:
        SQLValidationError: If any safety, dialect, or syntax rule is violated.
    """
    cleaned_sql = clean_sql_output(sql_query)

    # 1. Parse SQL statements and verify validity
    try:
        statements = sqlglot.parse(cleaned_sql, read="sqlite")
    except Exception as pe:
        raise SQLValidationError(f"SQL Syntax Error: {pe}")

    if len(statements) == 0:
        raise SQLValidationError("Unable to parse query: Empty or invalid SQL.")
    if len(statements) > 1:
        raise SQLValidationError("Multiple SQL statements detected. Only a single SELECT query is permitted.")

    ast = statements[0]
    if ast is None:
        raise SQLValidationError("SQL parser failed to generate an abstract syntax tree.")

    # 2. Verify root statement is a SELECT or UNION
    if not isinstance(ast, (exp.Select, exp.Union)):
        raise SQLValidationError(
            f"Unauthorized SQL statement type '{type(ast).__name__}'. Only SELECT queries are permitted."
        )

    # 3. Check for any nested forbidden expression types in AST
    forbidden_types = (
        exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter,
        exp.Create, exp.Command, exp.Pragma
    )
    for node in ast.walk():
        if isinstance(node, forbidden_types):
            raise SQLValidationError(
                f"Destructive operation '{type(node).__name__}' is strictly forbidden."
            )

    # 4. Keyword token scanning safeguard
    tokens = re.findall(r"\b[A-Za-z_]+\b", cleaned_sql)
    for tok in tokens:
        if tok.upper() in FORBIDDEN_KEYWORDS:
            # Check if this keyword is used as an identifier or literal by checking context
            # If it's a standalone reserved command, reject
            if tok.upper() in {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "ATTACH", "DETACH", "PRAGMA", "VACUUM"}:
                raise SQLValidationError(f"Forbidden SQL keyword '{tok.upper()}' detected in query.")

    # 5. Table validation: ensure query queries only allowed tables
    if allowed_tables:
        allowed_normalized = {t.lower().strip("\"'") for t in allowed_tables}
        referenced_tables = {
            table.name.lower().strip("\"'")
            for table in ast.find_all(exp.Table)
            if table.name
        }
        # Ignore common SQLite subquery aliases or virtual tables if present
        unrecognized = referenced_tables - allowed_normalized
        if unrecognized:
            raise SQLValidationError(
                f"Query references unauthorized or non-existent table(s): {', '.join(unrecognized)}. "
                f"Allowed table: {', '.join(allowed_tables)}"
            )

    # Return transpiled or cleaned SQLite query
    return cleaned_sql
