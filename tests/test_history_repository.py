"""Tests for history repository backends (In-memory and SQLite)."""

import tempfile
from pathlib import Path
import pytest

from src.history_repository import (
    InMemoryHistoryRepository,
    SQLiteHistoryRepository,
    get_history_repository,
)
from src.schemas import InferenceResult, ImageMetadata, PathologyPrediction


@pytest.fixture
def dummy_result():
    meta = ImageMetadata(
        filename="scan1.png",
        format="PNG",
        width=512,
        height=512,
        channels=1,
        size_bytes=45000,
    )
    pred = PathologyPrediction(
        pathology="Cardiomegaly",
        calibrated_score=0.72,
        probability=0.35,
        raw_logit=0.15,
        operating_threshold=0.05,
        is_detected=True,
    )
    return InferenceResult(
        analysis_id="ana_123456789abc",
        timestamp="2026-10-12T12:00:00Z",
        model_name="TorchXRayVision DenseNet-121",
        model_weights="densenet121-res224-all",
        device="cpu",
        inference_time_ms=65.2,
        predictions=[pred],
        top_findings=[pred],
        image_metadata=meta,
    )


def test_in_memory_repository(dummy_result):
    repo = InMemoryHistoryRepository()
    assert repo.count() == 0

    repo.add(dummy_result)
    assert repo.count() == 1

    records = repo.get_all()
    assert len(records) == 1
    assert records[0].analysis_id == dummy_result.analysis_id
    assert records[0].top_findings[0].pathology == "Cardiomegaly"

    repo.clear()
    assert repo.count() == 0
    assert len(repo.get_all()) == 0


def test_sqlite_repository(dummy_result):
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        repo = SQLiteHistoryRepository(db_path=tmp.name)
        assert repo.count() == 0

        repo.add(dummy_result)
        assert repo.count() == 1

        records = repo.get_all()
        assert len(records) == 1
        assert records[0].analysis_id == dummy_result.analysis_id
        assert records[0].top_findings[0].pathology == "Cardiomegaly"
        assert records[0].inference_time_ms == 65.2

        repo.clear()
        assert repo.count() == 0


def test_repository_factory():
    repo_mem = get_history_repository("in_memory")
    assert isinstance(repo_mem, InMemoryHistoryRepository)

    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        repo_sql = SQLiteHistoryRepository(db_path=tmp.name)
        assert isinstance(repo_sql, SQLiteHistoryRepository)
