import copy
from practice.rag_filter import filter_retrieved_chunks

def test_filter_retrieved_chunks_success():
    chunks = [
        {"source": "a.txt", "score": 0.65},
        {"source": "b.txt", "score": 0.30},
        {"source": "c.txt", "score": 0.45}
    ]

    result = filter_retrieved_chunks(
        chunks,
        min_score=0.45
    )

    assert len(result) == 2
    assert result[0]["source"] == "a.txt"
    assert result[1]["source"] == "c.txt"


def test_filter_all_below_threshold():
    chunks = [
        {"source": "a.txt", "score": 0.28},
        {"source": "b.txt", "score": 0.22}
    ]

    result = filter_retrieved_chunks(
        chunks,
        min_score=0.45
    )

    assert result == []



def test_filter_empty_chunks():
    chunks = []
    result = filter_retrieved_chunks(chunks, min_score=0.45)
    assert result == []


def test_filter_does_not_modify_input():
    chunks = [
        {"source": "a.txt", "score": 0.50},
        {"source": "b.txt", "score": 0.30}
    ]
    original = copy.deepcopy(chunks)
    result = filter_retrieved_chunks(chunks, min_score=0.45)
    assert chunks == original
    assert len(result) == 1
    assert result[0]["source"] == "a.txt"
