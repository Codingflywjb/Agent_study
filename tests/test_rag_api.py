from unittest.mock import MagicMock

import pytest
from sentence_transformers import SentenceTransformer

from app.config import RagSettings
from practice.llm_client import (
    LLMTimeoutError,
    LLMConnectionError
)
from fastapi.testclient import TestClient
from openai import APITimeoutError,APIConnectionError

from app import api

@pytest.fixture
def rag_settings_override(tmp_path):
    index_path = tmp_path / "rag_index.json"
    index_path.write_text("{}", encoding="utf-8")

    settings = RagSettings(
        embedding_model_name="BAAI/bge-small-zh-v1.5",
        index_path=index_path,
        min_score=0.60
    )
    api.app.dependency_overrides[api.get_rag_settings] = lambda: settings
    try:
       yield settings
    finally:
        api.app.dependency_overrides.pop(
            api.get_rag_settings,
            None
        )


def test_rag_ask_success(monkeypatch, rag_settings_override):
    index_path = rag_settings_override.index_path
    # 2. 模拟模型和 RAG 问答
    fake_model = object()

    mock_model = MagicMock(return_value=fake_model)

    mock_rag = MagicMock(
        return_value={
            "answer": "done 事件表示正常结束",
            "sources": ["sse.txt"]
        }
    )

    monkeypatch.setattr(
        api,
        "get_embedding_model",
        mock_model
    )

    monkeypatch.setattr(
        api,
        "ask_rag",
        mock_rag
    )

    # 3. 模拟 HTTP 请求
    with TestClient(api.app) as client:
        response = client.post(
            "/rag/ask",
            json={
                 "question": "SSE 的 done 事件表示什么？",
                "k": 2
            }
        )

    # 4. 验证结果
    assert response.status_code == 200

    assert response.json() == {
        "answer": "done 事件表示正常结束",
        "sources": ["sse.txt"]
    }

    mock_model.assert_called_once_with("BAAI/bge-small-zh-v1.5")

    mock_rag.assert_called_once_with(
        question="SSE 的 done 事件表示什么？",
        index_path=index_path,
        embedding_model=fake_model,
        k=2,
        min_score= 0.60
    )


def test_rag_ask_empty_question(monkeypatch, tmp_path):
    index_path = tmp_path / "rag_index.json"
    index_path.write_text("{}", encoding="utf-8")

    fake_settings = RagSettings(
        embedding_model_name="BAAI/bge-small-zh-v1.5",
        index_path=index_path,
        min_score=0.60
    )

    api.app.dependency_overrides[api.get_rag_settings] = lambda: fake_settings


    fake_model = object()

    mock_model = MagicMock(return_value=fake_model)

    mock_rag = MagicMock(
        return_value={
            "answer": "done 事件表示正常结束",
            "sources": ["sse.txt"]
        }
    )

    monkeypatch.setattr(
        api,
        "get_embedding_model",
        mock_model
    )

    monkeypatch.setattr(
        api,
        "ask_rag",
        mock_rag
    )
    try:
        with TestClient(api.app) as client:
            response = client.post(
                "/rag/ask",
                json={
                    "question": "    ",
                    "k": 2
                }
            )
    finally:
        api.app.dependency_overrides.pop(
            api.get_rag_settings,
            None
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "问题不能为空"
    mock_model.assert_not_called()
    mock_rag.assert_not_called()

def test_rag_ask_missing_index(monkeypatch, tmp_path):
    index_path = tmp_path / "not_exist.json"

    fake_settings = RagSettings(
        embedding_model_name="BAAI/bge-small-zh-v1.5",
        index_path=index_path,
        min_score=0.60
    )

    api.app.dependency_overrides[api.get_rag_settings] = lambda: fake_settings

    fake_model = object()

    mock_model = MagicMock(return_value=fake_model)

    mock_rag = MagicMock(
        return_value={
            "answer": "done 事件表示正常结束",
            "sources": ["sse.txt"]
        }
    )

    monkeypatch.setattr(
        api,
        "get_embedding_model",
        mock_model
    )

    monkeypatch.setattr(
        api,
        "ask_rag",
        mock_rag
    )
    try:
        with TestClient(api.app) as client:
            response = client.post(
                "/rag/ask",
                json={
                    "question": "SSE 的 done 事件表示什么？",
                    "k": 2
                }
            )
    finally:
        api.app.dependency_overrides.pop(
            api.get_rag_settings,
            None
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "文件不存在"
    mock_model.assert_not_called()
    mock_rag.assert_not_called()
    assert not index_path.exists()


def test_rag_ask_invalid_k(monkeypatch, rag_settings_override):
    mock_rag = MagicMock(
        return_value={
            "answer": "done 事件表示正常结束",
            "sources": ["sse.txt"]
        }
    )
    mock_model = MagicMock()
    monkeypatch.setattr(
        api,
        "get_embedding_model",
        mock_model
    )

    monkeypatch.setattr(
        api,
        "ask_rag",
        mock_rag
    )
    with TestClient(api.app) as client:
        response = client.post(
            "/rag/ask",
            json={
                "question": "SSE 的 done 事件表示什么？",
                "k": 0
            }
        )

    assert response.status_code == 422
    mock_model.assert_not_called()
    mock_rag.assert_not_called()

def test_rag_ask_model_load_failure(monkeypatch, tmp_path):
    rag_index_path = tmp_path / "rag_index.json"
    rag_index_path.write_text("{}", encoding="utf-8")

    fake_settings = RagSettings(
        embedding_model_name="BAAI/bge-small-zh-v1.5",
        index_path=rag_index_path,
        min_score=0.60
    )

    api.app.dependency_overrides[api.get_rag_settings] = lambda: fake_settings


    mock_model = MagicMock(side_effect=RuntimeError("模型加载失败"))

    mock_rag = MagicMock(
        return_value={
            "answer": "done 事件表示正常结束",
            "sources": ["sse.txt"]
        }
    )

    monkeypatch.setattr(
        api,
        "get_embedding_model",
        mock_model
    )

    monkeypatch.setattr(
        api,
        "ask_rag",
        mock_rag
    )
    try:
        with TestClient(api.app) as client:
            response = client.post(
                "/rag/ask",
                json={
                    "question": "SSE 的 done 事件表示什么？",
                    "k": 2
                }
            )
    finally:
        api.app.dependency_overrides.pop(
            api.get_rag_settings,
            None
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding 模型暂时不可用"
    mock_rag.assert_not_called()
    mock_model.assert_called_once_with("BAAI/bge-small-zh-v1.5")

def test_rag_ask_llm_timeout(monkeypatch, tmp_path):
    index_path = tmp_path / "rag_index.json"
    index_path.write_text("{}", encoding="utf-8")
    fake_settings = RagSettings(
        embedding_model_name="BAAI/bge-small-zh-v1.5",
        index_path=index_path,
        min_score=0.60
    )
    api.app.dependency_overrides[api.get_rag_settings] = lambda: fake_settings
    fake_model = object()
    mock_model = MagicMock(return_value=fake_model)
    mock_rag = MagicMock(
        side_effect=LLMTimeoutError("Deepseek请求超时")
    )
    monkeypatch.setattr(
        api,
        "get_embedding_model",
        mock_model
    )
    monkeypatch.setattr(
        api,
        "ask_rag",
        mock_rag
    )
    try:
        with TestClient(api.app) as client:
            response = client.post(
                "/rag/ask",
                json={
                    "question": "SSE 的 done 事件表示什么？",
                        "k": 2
                }
            )
    finally:
        api.app.dependency_overrides.pop(
            api.get_rag_settings,
            None
        )


    assert response.status_code == 504
    assert response.json()["detail"] == "大模型服务响应超时"

    mock_model.assert_called_once_with("BAAI/bge-small-zh-v1.5")
    mock_rag.assert_called_once_with(
        question="SSE 的 done 事件表示什么？",
        index_path=index_path,
        embedding_model=fake_model,
        k=2,
        min_score=0.60
    )


def test_rag_ask_llm_connection_error(monkeypatch, tmp_path):
    temp_path = tmp_path / "rag_index.json"
    temp_path.write_text("{}", encoding="utf-8")
    fake_settings = RagSettings(
        embedding_model_name="BAAI/bge-small-zh-v1.5",
        index_path=temp_path,
        min_score=0.60
    )
    api.app.dependency_overrides[api.get_rag_settings] = lambda: fake_settings
    fake_model = object()
    mock_model = MagicMock(return_value=fake_model)
    mock_rag = MagicMock(side_effect=LLMConnectionError("Deepseek连接超时"))
    monkeypatch.setattr(
        api,
        "get_embedding_model",
        mock_model
    )
    monkeypatch.setattr(
        api,
        "ask_rag",
        mock_rag
    )
    try:
        with TestClient(api.app) as client:
            response = client.post(
                "/rag/ask",
                json={
                    "question": "SSE 的 done 事件表示什么？",
                    "k": 2
                }
            )
    finally:
        api.app.dependency_overrides.pop(
            api.get_rag_settings,
            None
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "大模型服务暂时不可用"
    mock_rag.assert_called_once()

def test_embedding_model_cache(monkeypatch):
    fake_model_a = object()
    fake_model_b = object()

    mock_constructor = MagicMock(side_effect=[fake_model_a, fake_model_b])

    monkeypatch.setattr(
        api,
        "SentenceTransformer",
        mock_constructor
    )

    api.get_embedding_model.cache_clear()

    try:
        result_a1=api.get_embedding_model("fake_model_a")
        result_a2=api.get_embedding_model("fake_model_a")
        result_a3=api.get_embedding_model("fake_model_b")
        assert result_a1 is result_a2
        assert result_a1 is not result_a3
        assert mock_constructor.call_count == 2
        mock_constructor.assert_any_call("fake_model_a", device="cpu")
        mock_constructor.assert_any_call("fake_model_b", device="cpu")
    finally:
        api.get_embedding_model.cache_clear()