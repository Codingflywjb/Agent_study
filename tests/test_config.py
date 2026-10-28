import pytest

from app.config import load_rag_settings,PROJECT_ROOT


def test_rag_settings_min_score(monkeypatch):
    monkeypatch.setenv("RAG_MIN_SCORE", "0.65")

    settings = load_rag_settings()

    assert settings.min_score == 0.65
    assert isinstance(settings.min_score, float)

def test_rag_settings_relative_path(monkeypatch):
    monkeypatch.setenv("RAG_INDEX_PATH", "data/test_index.json")

    settings = load_rag_settings()

    assert settings.index_path == PROJECT_ROOT / "data/test_index.json"

def test_rag_settings_absolute_path(monkeypatch, tmp_path):
    absolute_path = tmp_path / "custom_index.json"
    monkeypatch.setenv("RAG_INDEX_PATH",str(absolute_path))
    settings = load_rag_settings()
    assert settings.index_path == absolute_path

def test_rag_settings_invalid_score(monkeypatch):
    monkeypatch.setenv("RAG_MIN_SCORE", "1.5")
    with pytest.raises(ValueError):
        load_rag_settings()

def test_rag_settings_invalid_score_type(monkeypatch):
    monkeypatch.setenv("RAG_MIN_SCORE", "abc")

    with pytest.raises(ValueError):
        load_rag_settings()

