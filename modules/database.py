"""
Database Module
Manages SQLite database creation, schema introspection, and connection lifecycle.
"""

from typing import Optional, List, Dict, Any, Tuple
import sqlite3
import pandas as pd


class DatabaseError(Exception):
    """Custom exception for database connection or schema errors."""
    pass


class SQLiteManager:
    """
    Manages SQLite database lifecycle for loaded DataFrames.
    Supports in-memory or on-disk database instances.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None

    def get_connection(self) -> sqlite3.Connection:
        """Returns active SQLite connection or creates a new one."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._connection

    def load_dataframe(self, df: pd.DataFrame, table_name: str = "data_table") -> int:
        """
        Stores the given DataFrame into SQLite.
        Replaces the table if it already exists.

        Returns:
            Number of rows loaded.
        """
        if df is None or df.empty:
            raise DatabaseError("Cannot load empty DataFrame into SQLite.")

        conn = self.get_connection()
        try:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            return len(df)
        except Exception as exc:
            raise DatabaseError(f"Failed to load DataFrame into SQLite: {exc}")

    def get_table_schema_ddl(self, table_name: str = "data_table") -> str:
        """
        Generates an exact DDL representation (CREATE TABLE ...) using PRAGMA table_info.
        This provides high-fidelity schema representation for the LLM.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns_info = cursor.fetchall()
            if not columns_info:
                raise DatabaseError(f"Table '{table_name}' does not exist or has no columns.")

            col_defs: List[str] = []
            for col in columns_info:
                # col format: (cid, name, type, notnull, dflt_value, pk)
                col_name = col[1]
                col_type = col[2] or "TEXT"
                col_defs.append(f"    \"{col_name}\" {col_type}")

            ddl = f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n);"
            return ddl
        except Exception as exc:
            raise DatabaseError(f"Failed to generate schema DDL for '{table_name}': {exc}")

    def get_sample_records(self, table_name: str = "data_table", limit: int = 3) -> List[Dict[str, Any]]:
        """Fetches up to `limit` sample rows from the table as dictionaries."""
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute(f"SELECT * FROM \"{table_name}\" LIMIT {int(limit)};")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            raise DatabaseError(f"Failed to fetch sample records: {exc}")
        finally:
            conn.row_factory = None

    def get_row_count(self, table_name: str = "data_table") -> int:
        """Returns the total number of rows in the table."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM \"{table_name}\";")
            res = cursor.fetchone()
            return res[0] if res else 0
        except Exception as exc:
            raise DatabaseError(f"Failed to retrieve row count: {exc}")

    def close(self) -> None:
        """Closes the active database connection if open."""
        if self._connection:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None
