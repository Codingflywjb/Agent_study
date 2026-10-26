import pytest
import numpy as np
from unittest.mock import MagicMock
from practice.rag_embedding import QUERY_PREFIX
from practice.rag_search import search_knowledge



def test_search_knowledge_success(tmp_path):
    file1 = tmp_path / "a.txt"
    file1.write_text("AAAA", encoding="utf-8")

    file2 = tmp_path / "b.txt"
    file2.write_text("BBBB", encoding="utf-8")

    model = MagicMock()

    def fake_encode(text, normalize_embeddings=True):
        if text == "AAAA":
            return np.array([1.0, 0.0])

        if text == "BBBB":
            return np.array([0.0, 1.0])

        if text.startswith(QUERY_PREFIX):
            return np.array([1.0, 0.0])

        raise ValueError("Unexpected text")


    model.encode.side_effect = fake_encode

    result = search_knowledge(
            query="查询问题",
            directory=tmp_path,
            model=model,
            k=1,
            chunk_size=4,
            overlap=0
        )

    assert len(result) == 1
    assert result[0]["source"] == "a.txt"
    assert result[0]["score"] == pytest.approx(1.0)
    model.encode.assert_called_with(QUERY_PREFIX+"查询问题",normalize_embeddings=True)

def test_search_knowledge_invalid_query(tmp_path):
    model = MagicMock()
    file1 = tmp_path / "a.txt"
    file1.write_text("AAAA", encoding="utf-8")

    file2 = tmp_path / "b.txt"
    file2.write_text("BBBB", encoding="utf-8")

    with pytest.raises(ValueError):
        search_knowledge(
            query="",
            directory=tmp_path,
            model=model,
            k=1,
            chunk_size=4,
            overlap=0
        )

    with pytest.raises(ValueError):
        search_knowledge(
            query="  ",
            directory=tmp_path,
            model=model,
            k=1,
            chunk_size=4,
            overlap=0
        )

def test_search_knowledge_empty_directory(tmp_path):
    model = MagicMock()
    result = search_knowledge(
        query="查询问题",
        directory=tmp_path,
        model=model,
        k=1,
        chunk_size=4,
        overlap=0
    )
    assert result == []
    model.encode.assert_not_called()
