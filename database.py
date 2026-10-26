"""SQLite/PostgreSQL storage adapter for the Streamlit app."""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Any, Iterable


TABLES = {
    "news_history": """CREATE TABLE IF NOT EXISTS news_history (
        id {identity} PRIMARY KEY,
        symbol TEXT NOT NULL,
        captured_at TEXT,
        published_at TEXT,
        source TEXT,
        headline TEXT,
        url TEXT,
        sentiment DOUBLE PRECISION,
        relevance DOUBLE PRECISION,
        UNIQUE(symbol, headline, published_at)
    )""",
    "predictions": """CREATE TABLE IF NOT EXISTS predictions (
        id {identity} PRIMARY KEY,
        symbol TEXT NOT NULL,
        prediction_date TEXT,
        target_date TEXT,
        horizon INTEGER,
        latest_close DOUBLE PRECISION,
        predicted_close DOUBLE PRECISION,
        predicted_return DOUBLE PRECISION,
        signal TEXT,
        confidence DOUBLE PRECISION,
        bullish DOUBLE PRECISION,
        bearish DOUBLE PRECISION,
        low DOUBLE PRECISION,
        high DOUBLE PRECISION,
        market_condition TEXT,
        market_return DOUBLE PRECISION,
        sector_return DOUBLE PRECISION,
        news_score DOUBLE PRECISION,
        news_json TEXT,
        feature_json TEXT,
        model_version TEXT,
        actual_close DOUBLE PRECISION,
        actual_return DOUBLE PRECISION,
        percentage_error DOUBLE PRECISION,
        direction_correct INTEGER,
        evaluated_at TEXT,
        UNIQUE(symbol, prediction_date, horizon, model_version)
    )""",
    "model_registry": """CREATE TABLE IF NOT EXISTS model_registry (
        id {identity} PRIMARY KEY,
        symbol TEXT,
        horizon INTEGER,
        version TEXT,
        trained_at TEXT,
        rows INTEGER,
        metrics_json TEXT,
        active INTEGER
    )""",
}


def uses_postgres(database_url: str | None = None) -> bool:
    """Return whether the configured database is a PostgreSQL URL."""
    value = database_url if database_url is not None else os.getenv("DATABASE_URL", "")
    return value.startswith(("postgres://", "postgresql://"))


def database_backend(database_url: str | None = None) -> str:
    return "PostgreSQL (hosted)" if uses_postgres(database_url) else "SQLite (local file)"


class CursorAdapter:
    """Preserve the small sqlite cursor API used by app.py."""

    def __init__(self, cursor: Any, postgres: bool):
        self._cursor = cursor
        self._postgres = postgres

    @staticmethod
    def _postgres_sql(query: str) -> str:
        # Psycopg treats percent signs as placeholder markers even in SQL string
        # literals, so escape query literals before introducing its %s params.
        query = query.replace("INSERT OR IGNORE INTO", "INSERT INTO")
        replace_insert = "INSERT OR REPLACE INTO" in query
        query = query.replace("INSERT OR REPLACE INTO", "INSERT INTO")
        query = query.replace("?", "\x00PARAM\x00").replace("%", "%%").replace("\x00PARAM\x00", "%s")
        if "INSERT INTO news_history" in query and "ON CONFLICT" not in query:
            query += " ON CONFLICT (symbol, headline, published_at) DO NOTHING"
        elif replace_insert and "INSERT INTO predictions" in query:
            match = re.search(r"INSERT INTO predictions\s*\(([^)]+)\)", query, re.IGNORECASE)
            if match:
                columns = [column.strip() for column in match.group(1).split(",")]
                updates = [
                    f"{column} = EXCLUDED.{column}"
                    for column in columns
                    if column not in {"symbol", "prediction_date", "horizon", "model_version"}
                ]
                query += (
                    " ON CONFLICT (symbol, prediction_date, horizon, model_version) DO UPDATE SET "
                    + ", ".join(updates)
                )
        return query

    def execute(self, query: str, parameters: Iterable[Any] = ()) -> "CursorAdapter":
        self._cursor.execute(self._postgres_sql(query) if self._postgres else query, tuple(parameters))
        return self

    def executemany(self, query: str, parameters: Iterable[Iterable[Any]]) -> "CursorAdapter":
        self._cursor.executemany(
            self._postgres_sql(query) if self._postgres else query,
            [tuple(row) for row in parameters],
        )
        return self

    def fetchone(self) -> Any:
        return self._cursor.fetchone()

    def fetchall(self) -> list[Any]:
        return self._cursor.fetchall()

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount


class ConnectionAdapter:
    """Expose sqlite-like execute methods for either supported backend."""

    def __init__(self, connection: Any, postgres: bool):
        self._connection = connection
        self._postgres = postgres

    def execute(self, query: str, parameters: Iterable[Any] = ()) -> CursorAdapter:
        cursor = self._connection.cursor()
        return CursorAdapter(cursor, self._postgres).execute(query, parameters)

    def executemany(self, query: str, parameters: Iterable[Iterable[Any]]) -> CursorAdapter:
        cursor = self._connection.cursor()
        return CursorAdapter(cursor, self._postgres).executemany(query, parameters)

    def commit(self) -> None:
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


def connect_database(database_url: str | None = None, sqlite_path: str | None = None) -> ConnectionAdapter:
    """Connect to hosted PostgreSQL or the existing local SQLite file."""
    url = database_url if database_url is not None else os.getenv("DATABASE_URL", "")
    if uses_postgres(url):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("PostgreSQL support requires psycopg; install requirements.txt") from exc
        return ConnectionAdapter(psycopg.connect(url, row_factory=dict_row), postgres=True)
    path = sqlite_path or os.getenv("APP_DB_PATH", "stock_ai.db")
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return ConnectionAdapter(connection, postgres=False)


def initialize_database(database_url: str | None = None, sqlite_path: str | None = None) -> None:
    """Create the app schema in the selected backend."""
    postgres = uses_postgres(database_url)
    connection = connect_database(database_url, sqlite_path)
    identity = "BIGSERIAL" if postgres else "INTEGER"
    try:
        for statement in TABLES.values():
            connection.execute(statement.format(identity=identity))
        connection.commit()
    finally:
        connection.close()