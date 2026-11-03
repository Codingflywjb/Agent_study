from unittest import result

import pytest
import numpy as np
from unittest.mock import MagicMock

from practice import rag_cached_search
from practice.rag_index import save_vector_index
from practice.rag_embedding import QUERY_PREFIX
from practice.rag_cached_search import search_cached_knowledge,search_cached_knowledge_v2

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


def test_search_cached_knowledge_v2_success(monkeypatch):
    fake_model = object()
    fake_embedding = [0.1,0.2,0.3]
    fake_chunks=[
        {
            "source": "sse.txt",
            "chunk_id": 0,
            "content":"SSE 的 done 事件表示正常结束",
            "embedding": fake_embedding
        }
    ]
    fake_results=[
        {
            "source": "sse.txt",
            "chunk_id": 0,
            "content":"SSE 的 done 事件表示正常结束",
            "score": 0.95
        }
    ]
    mock_embed = MagicMock(return_value=fake_embedding)
    mock_load = MagicMock(return_value=fake_chunks)
    mock_topk = MagicMock(return_value=fake_results)
    monkeypatch.setattr(rag_cached_search,"embed_text",mock_embed)
    monkeypatch.setattr(rag_cached_search,"load_vector_index_v2",mock_load)
    monkeypatch.setattr(rag_cached_search,"top_k_search",mock_topk)
    result=search_cached_knowledge_v2(
        query="什么是SSE？",
        index_path="test_index.json",
        model=fake_model,
        expected_model_name="test_model",
        k=2
    )
    mock_load.assert_called_once_with(
        index_path="test_index.json",
        expected_model_name="test_model",
        expected_dim=3
    )
    assert result == fake_results
    mock_topk.assert_called_once_with(
        query_embedding=fake_embedding,
        chunks=fake_chunks,
        k=2
    )
    mock_embed.assert_called_once_with(
        text="什么是SSE？",
        model=fake_model,
        is_query=True
    )

def test_search_cached_knowledge_v2_empty_index(monkeypatch):
    fake_model = object()

    mock_embed = MagicMock(return_value=[0.1,0.2,0.3])
    mock_load = MagicMock(return_value=[])
    mock_topk = MagicMock()
    monkeypatch.setattr(rag_cached_search,"embed_text",mock_embed)
    monkeypatch.setattr(rag_cached_search,"load_vector_index_v2",mock_load)
    monkeypatch.setattr(rag_cached_search,"top_k_search",mock_topk)

    result=rag_cached_search.search_cached_knowledge_v2(
        query="什么是SSE？",
        index_path="test_index.json",
        model=fake_model,
        expected_model_name="test_model",
        k=2
    )
    assert result == []

    mock_embed.assert_called_once_with(
        text="什么是SSE？",
        model=fake_model,
        is_query=True
    )

    mock_load.assert_called_once_with(
        index_path="test_index.json",
        expected_model_name="test_model",
        expected_dim=3
    )

    mock_topk.assert_not_called()


def test_search_cached_knowledge_v2_invalid_query():
    with pytest.raises(ValueError,match="Query is required"):
        search_cached_knowledge_v2(
            query="   ",
            index_path="test_index.json",
            model=MagicMock(),
            expected_model_name="test_model",
            k=2
        )


def test_search_cached_knowledge_v2_model_mismatch(monkeypatch):
    fake_model = object()

    mock_embed = MagicMock(
        return_value=[0.1, 0.2, 0.3]
    )

    mock_load = MagicMock(side_effect=ValueError("Invalid embedding_model_name"))
    mock_topk = MagicMock()

    monkeypatch.setattr(
        rag_cached_search,
        "embed_text",
        mock_embed
    )

    monkeypatch.setattr(rag_cached_search,"load_vector_index_v2",mock_load)
    monkeypatch.setattr(
        rag_cached_search,
        "top_k_search",
        mock_topk
    )
    with pytest.raises(ValueError,match="Invalid embedding_model_name"):
        search_cached_knowledge_v2(
            query="什么是SSE？",
            index_path="test_index.json",
            model=fake_model,
            expected_model_name="model-B",
            k=2
        )
    mock_load.assert_called_once_with(
        index_path="test_index.json",
        expected_model_name="model-B",
        expected_dim=3
    )

    mock_topk.assert_not_called()
