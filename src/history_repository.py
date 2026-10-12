"""Modular history repository interface and implementations for demo tracking.

By default, an in-memory repository is used to keep the POC stateless,
avoiding permanent storage of medical metadata while remaining fully modular
for future database integration (e.g. SQLite, PostgreSQL).
"""

from abc import ABC, abstractmethod
import json
import sqlite3
from typing import List, Optional
from pathlib import Path

from src.schemas import InferenceResult


class BaseHistoryRepository(ABC):
    """Abstract base class for analysis history repositories."""

    @abstractmethod
    def add(self, result: InferenceResult) -> None:
        """Store an inference execution record."""
        pass

    @abstractmethod
    def get_all(self, limit: int = 50) -> List[InferenceResult]:
        """Retrieve recent inference execution records."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear stored execution records."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total count of stored records."""
        pass


class InMemoryHistoryRepository(BaseHistoryRepository):
    """Stateless in-memory repository for demo session history.

    Stores results exclusively in RAM for the active session.
    Never persists records to disk and never stores uploaded image data.
    """

    def __init__(self) -> None:
        self._records: List[InferenceResult] = []

    def add(self, result: InferenceResult) -> None:
        # Avoid storing any large raw buffers; InferenceResult only stores metadata & scores
        self._records.append(result)

    def get_all(self, limit: int = 50) -> List[InferenceResult]:
        # Return newest first
        return list(reversed(self._records[-limit:]))

    def clear(self) -> None:
        self._records.clear()

    def count(self) -> int:
        return len(self._records)


class SQLiteHistoryRepository(BaseHistoryRepository):
    """Lightweight SQLite persistence repository for optional local storage."""

    def __init__(self, db_path: str = "data/analysis_history.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
                    analysis_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_weights TEXT NOT NULL,
                    device TEXT NOT NULL,
                    inference_time_ms REAL NOT NULL,
                    top_finding TEXT NOT NULL,
                    top_score REAL NOT NULL,
                    record_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def add(self, result: InferenceResult) -> None:
        top_finding_name = result.top_findings[0].pathology if result.top_findings else "None"
        top_finding_score = result.top_findings[0].calibrated_score if result.top_findings else 0.0

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO analysis_history (
                    analysis_id, timestamp, model_name, model_weights,
                    device, inference_time_ms, top_finding, top_score, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.analysis_id,
                    result.timestamp,
                    result.model_name,
                    result.model_weights,
                    result.device,
                    result.inference_time_ms,
                    top_finding_name,
                    top_finding_score,
                    result.model_dump_json(),
                ),
            )
            conn.commit()

    def get_all(self, limit: int = 50) -> List[InferenceResult]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT record_json FROM analysis_history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            return [InferenceResult.model_validate_json(row[0]) for row in rows]

    def clear(self) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM analysis_history")
            conn.commit()

    def count(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM analysis_history")
            return int(cursor.fetchone()[0])


def get_history_repository(backend: str = "in_memory") -> BaseHistoryRepository:
    """Factory creating appropriate history repository instance.

    Args:
        backend: 'in_memory' for stateless demo or 'sqlite' for local persistence.

    Returns:
        Configured BaseHistoryRepository.
    """
    if backend == "sqlite":
        return SQLiteHistoryRepository()
    return InMemoryHistoryRepository()
