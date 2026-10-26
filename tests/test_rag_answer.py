import pytest
from unittest.mock import MagicMock

from practice import rag_answer

def test_ask_rag_empty_results(monkeypatch):
    mock_search = MagicMock(return_value=[])
    mock_llm = MagicMock()

    monkeypatch.setattr(
        rag_answer,
        "search_cached_knowledge",
        mock_search
    )

    monkeypatch.setattr(
        rag_answer,
        "ask_llm",
        mock_llm
    )

    result = rag_answer.ask_rag(
        question="FastAPI 如何实现流式输出？",
        index_path="test_index.json",
        embedding_model=MagicMock(),
        k=2
    )

    assert result == {
        "answer": "知识库中没有可用的参考资料，暂时无法回答该问题。",
        "sources": []
    }

    mock_llm.assert_not_called()

def test_ask_rag_success(monkeypatch):
    retrieved_chunks = [
        {
            "source": "fastapi.txt",
            "chunk_id": 0,
            "content": "FastAPI 支持 StreamingResponse",
            "score": 0.95
        },
        {
            "source": "fastapi.txt",
            "chunk_id": 1,
            "content": "可以配合生成器逐步返回内容",
            "score": 0.90
        },
        {
            "source": "sse.txt",
            "chunk_id": 0,
            "content": "SSE 可以持续发送事件",
            "score": 0.85
        }
    ]

    mock_search = MagicMock(return_value=retrieved_chunks)

    mock_prompt = MagicMock(return_value="模拟的 RAG Prompt")

    mock_llm = MagicMock(
        return_value="FastAPI 可以通过 StreamingResponse 实现流式输出。"
    )

    monkeypatch.setattr(
        rag_answer,
        "search_cached_knowledge",
        mock_search
    )

    monkeypatch.setattr(
        rag_answer,
        "build_rag_prompt",
        mock_prompt
    )

    monkeypatch.setattr(
        rag_answer,
        "ask_llm",
        mock_llm
    )

    embedding_model = MagicMock()

    result = rag_answer.ask_rag(
        question="FastAPI 如何实现流式输出？",
        index_path="test_index.json",
        embedding_model=embedding_model,
        k=3
    )

    assert result["answer"] == (
        "FastAPI 可以通过 StreamingResponse 实现流式输出。"
    )

    assert result["sources"] == [
        "fastapi.txt",
        "sse.txt"
    ]

    mock_llm.assert_called_once_with("模拟的 RAG Prompt")

    mock_search.assert_called_once_with(
        query="FastAPI 如何实现流式输出？",
        index_path="test_index.json",
        model=embedding_model,
        k=3
    )

def test_ask_rag_invalid_question():
    with pytest.raises(ValueError):
        rag_answer.ask_rag(
            question="",
            index_path="test_index.json",
            embedding_model=MagicMock(),
            k=2
        )
    with pytest.raises(ValueError):
        rag_answer.ask_rag(
            question="    ",
            index_path="test_index.json",
            embedding_model=MagicMock(),
            k=2
        )

def test_ask_rag_invalid_k():
    with pytest.raises(ValueError):
        rag_answer.ask_rag(
            question="FastAPI 如何实现流式输出？",
            index_path="test_index.json",
            embedding_model=MagicMock(),
            k=0
        )

    with pytest.raises(ValueError):
        rag_answer.ask_rag(
            question="FastAPI 如何实现流式输出？",
            index_path="test_index.json",
            embedding_model=MagicMock(),
            k=-1
        )

def test_ask_rag_low_similarity(monkeypatch):
    retrieved_chunks = [
        {
            "source": "sse.txt",
            "chunk_id": 0,
            "content": "SSE 支持服务器推送事件",
            "score": 0.2806
        },
        {
            "source": "fastapi.txt",
            "chunk_id": 0,
            "content": "FastAPI 支持 StreamingResponse",
            "score": 0.2201
        }
    ]
    mock_search = MagicMock(return_value=retrieved_chunks)
    mock_prompt = MagicMock()
    mock_llm = MagicMock()

    monkeypatch.setattr(
        rag_answer,
        "search_cached_knowledge",
        mock_search
    )
    monkeypatch.setattr(
        rag_answer,
        "build_rag_prompt",
        mock_prompt
    )
    monkeypatch.setattr(
        rag_answer,
        "ask_llm",
        mock_llm
    )
    result = rag_answer.ask_rag(
        question="唐朝是哪一年建立的？",
        index_path="test_index.json",
        embedding_model=MagicMock(),
        k=2,
        min_score=0.45
    )
    assert result["sources"] == []
    assert "没有足够相关的参考资料" in result["answer"]
    mock_prompt.assert_not_called()
    mock_search.assert_called_once()
    mock_llm.assert_not_called()