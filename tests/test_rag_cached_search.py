import pytest
import numpy as np
from unittest.mock import MagicMock

from practice.rag_index import save_vector_index
from practice.rag_embedding import QUERY_PREFIX
from practice.rag_cached_search import search_cached_knowledge

def test_cached_search_success(tmp_path):
    indexed_chunks = [
        {
            "source": "fastapi.txt",
            "chunk_id": 0,
            "content": "FastAPI 支持流式输出",
            "embedding": [1.0, 0.0]
        },
        {
            "source": "python.txt",
            "chunk_id": 0,
            "content": "Python 支持异常处理",
            "embedding": [0.0, 1.0]
        }
    ]

    index_path = tmp_path / "index.json"
    save_vector_index(index_path=index_path,indexed_chunks=indexed_chunks)

    model = MagicMock()
    model.encode.return_value = np.array([1.0,0.0])

    question= "FastAPI 如何实现流式输出？"
    result=search_cached_knowledge(query=question,model=model,index_path=index_path,k=1)

    assert len(result)==1
    assert result[0]["source"]=="fastapi.txt"
    assert result[0]["score"]==pytest.approx(1.0)

    model.encode.assert_called_once_with(
        QUERY_PREFIX+question,
        normalize_embeddings=True
    )

def test_cached_search_invalid_query(tmp_path):
    indexed_chunks = [
        {
            "source": "fastapi.txt",
            "chunk_id": 0,
            "content": "FastAPI 支持流式输出",
            "embedding": [1.0, 0.0]
        },
        {
            "source": "python.txt",
            "chunk_id": 0,
            "content": "Python 支持异常处理",
            "embedding": [0.0, 1.0]
        }
    ]

    index_path = tmp_path / "index.json"
    save_vector_index(index_path=index_path, indexed_chunks=indexed_chunks)
    model = MagicMock()
    model.encode.return_value = np.array([1.0, 0.0])
    with pytest.raises(ValueError):
        search_cached_knowledge(query="",model=model,index_path=index_path,k=1)
    with pytest.raises(ValueError):
        search_cached_knowledge(query="    ", model=model, index_path=index_path, k=1)

    model.encode.assert_not_called()

def test_cached_search_invalid_k(tmp_path):
    indexed_chunks = [
        {
            "source": "fastapi.txt",
            "chunk_id": 0,
            "content": "FastAPI 支持流式输出",
            "embedding": [1.0, 0.0]
        },
        {
            "source": "python.txt",
            "chunk_id": 0,
            "content": "Python 支持异常处理",
            "embedding": [0.0, 1.0]
        }
    ]

    index_path = tmp_path / "index.json"
    save_vector_index(index_path=index_path, indexed_chunks=indexed_chunks)
    model = MagicMock()
    model.encode.return_value = np.array([1.0, 0.0])
    question = "FastAPI 如何实现流式输出？"
    with pytest.raises(ValueError):
        search_cached_knowledge(query=question,model=model,index_path=index_path,k=0)
    with pytest.raises(ValueError):
        search_cached_knowledge(query=question,model=model,index_path=index_path,k=-1)

    model.encode.assert_not_called()

def test_cached_search_empty_index(tmp_path):
    index_path = tmp_path / "index.json"
    save_vector_index(index_path=index_path,indexed_chunks=[])
    model = MagicMock()
    model.encode.return_value = np.array([1.0,0.0])
    question = "FastAPI 如何实现流式输出？"
    result=search_cached_knowledge(query=question,model=model,index_path=index_path,k=1)
    assert result == []
    model.encode.assert_not_called()

def test_cached_search_missing_index(tmp_path):
    index_path = tmp_path / "not_exist.json"
    model = MagicMock()

    assert not index_path.exists()

    question = "FastAPI 如何实现流式输出？"
    with pytest.raises(FileNotFoundError):
        search_cached_knowledge(query=question,model=model,index_path=index_path,k=1)

    model.encode.assert_not_called()