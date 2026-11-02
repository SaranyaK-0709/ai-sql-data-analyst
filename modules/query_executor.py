"""
Query Executor Module
Executes validated SQL against SQLite with engine-level read-only authorizer enforcement,
query timeouts, row truncation limits, and execution profiling.
"""

from typing import Tuple, Optional, Any
import time
import sqlite3
import pandas as pd


class QueryExecutionError(Exception):
    """Custom exception raised during SQL query execution."""
    pass


class QueryTimeoutError(QueryExecutionError):
    """Raised when query execution exceeds the allowed time limit."""
    pass


# Explicitly forbidden modifying actions at SQLite C-engine level
FORBIDDEN_SQLITE_ACTIONS = {
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_UPDATE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_DETACH,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_PRAGMA
}


def read_only_authorizer(action: int, arg1: Any, arg2: Any, db_name: Any, trigger_name: Any) -> int:
    """
    SQLite authorizer callback that enforces strict read-only execution at the SQLite C-engine level.
    Rejects any modifying operations regardless of how they are constructed.
    """
    if action in FORBIDDEN_SQLITE_ACTIONS:
        return sqlite3.SQLITE_DENY

    return sqlite3.SQLITE_OK


class QueryExecutor:
    """
    Safely executes validated queries against an SQLite connection.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        timeout_seconds: int = 10,
        max_rows: int = 5000
    ):
        self.conn = connection
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows

    def execute_query(self, sql_query: str) -> Tuple[pd.DataFrame, float, int, bool]:
        """
        Executes a validated read-only SQL query against SQLite.

        Returns:
            Tuple of:
            - DataFrame with query results
            - Execution time in milliseconds
            - Total rows returned
            - Truncated flag (True if rows exceeded max_rows)

        Raises:
            QueryExecutionError: If execution fails or is denied.
            QueryTimeoutError: If execution exceeds the allotted timeout.
        """
        # Attach the read-only authorizer to prevent any modifications at the engine level
        self.conn.set_authorizer(read_only_authorizer)

        start_time = time.perf_counter()
        cursor = self.conn.cursor()

        # Setup timeout watchdog using progress handler
        # Progress handler is called every N virtual machine instructions
        start_clock = time.time()
        timeout_limit = self.timeout_seconds

        def progress_callback() -> int:
            if time.time() - start_clock > timeout_limit:
                # Returning 1 interrupts the SQLite execution
                return 1
            return 0

        self.conn.set_progress_handler(progress_callback, 1000)

        try:
            cursor.execute(sql_query)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Fetch up to max_rows + 1 to detect if truncation occurred
            raw_rows = cursor.fetchmany(self.max_rows + 1)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            truncated = False
            if len(raw_rows) > self.max_rows:
                truncated = True
                rows = raw_rows[:self.max_rows]
            else:
                rows = raw_rows

            df = pd.DataFrame(rows, columns=columns)
            row_count = len(df)

            return df, round(duration_ms, 2), row_count, truncated

        except (sqlite3.OperationalError, sqlite3.DatabaseError) as op_err:
            err_text = str(op_err)
            if "interrupted" in err_text.lower():
                raise QueryTimeoutError(f"Query timed out after exceeding {self.timeout_seconds} seconds.")
            elif "not authorized" in err_text.lower():
                raise QueryExecutionError("Security Violation: Database rejected non-read-only operation.")
            else:
                raise QueryExecutionError(f"SQLite Operational Error: {err_text}")

        except Exception as exc:
            raise QueryExecutionError(f"Error executing SQL query: {exc}")

        finally:
            # Clear progress handler & authorizer
            self.conn.set_progress_handler(None, 0)
            self.conn.set_authorizer(None)
            cursor.close()
