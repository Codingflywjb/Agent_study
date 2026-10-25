import pytest
from practice.rag_retriever import top_k_search

def test_top_k_search():
    chunks = [
        {
            "source": "a.txt",
            "chunk_id": 0,
            "content": "A",
            "embedding": [1.0, 0.0]
        },
        {
            "source": "b.txt",
            "chunk_id": 0,
            "content": "B",
            "embedding": [0.0, 1.0]
        },
        {
            "source": "c.txt",
            "chunk_id": 0,
            "content": "C",
            "embedding": [0.8, 0.6]
        }
    ]

    result = top_k_search(
        [1.0, 0.0],
        chunks,
        k=2
    )

    assert len(result) == 2
    assert result[0]["source"] == "a.txt"
    assert result[0]["score"] == pytest.approx(1.0)

    assert result[1]["source"] == "c.txt"
    assert result[1]["score"] == pytest.approx(0.8)

def test_top_k_invalid_k():
    chunks = [
        {
            "source": "a.txt",
            "chunk_id": 0,
            "content": "A",
            "embedding": [1.0, 0.0]
        },
        {
            "source": "b.txt",
            "chunk_id": 0,
            "content": "B",
            "embedding": [0.0, 1.0]
        },
        {
            "source": "c.txt",
            "chunk_id": 0,
            "content": "C",
            "embedding": [0.8, 0.6]
        }
    ]

    with pytest.raises(ValueError):
         top_k_search(
            [1.0, 0.0],
            chunks,
            k=0
        )

    with pytest.raises(ValueError):
        top_k_search(
            [1.0, 0.0],
            chunks,
            k=-1
        )

def test_top_k_empty_chunks():
    chunks =[]
    result = top_k_search(
        [1.0, 0.0],
        chunks,
        k=2
    )
    assert result == []

def test_top_k_larger_than_chunks():
    chunks = [
        {
            "source": "a.txt",
            "chunk_id": 0,
            "content": "A",
            "embedding": [1.0, 0.0]
        },
        {
            "source": "b.txt",
            "chunk_id": 0,
            "content": "B",
            "embedding": [0.0, 1.0]
        }
    ]
    result = top_k_search(
        [1.0, 0.0],
        chunks,
        k=10
    )
    assert len(result) == 2
    assert result[0]["source"] == "a.txt"
    assert result[1]["source"] == "b.txt"

def test_top_k_does_not_modify_input():
    chunk = {
        "source": "a.txt",
        "chunk_id": 0,
        "content": "FastAPI",
        "embedding": [1.0, 0.0]
    }
    result = top_k_search(
        [1.0, 0.0],
        [chunk],
        k=1
    )
    assert "score" not in chunk
    assert "embedding" in chunk

    assert "score" in result[0]
    assert "embedding" not in result[0]