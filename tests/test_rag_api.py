from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app import api

def test_rag_ask_success(monkeypatch, tmp_path):
    # 1. 创建一个临时索引文件
    index_path = tmp_path / "rag_index.json"
    index_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        api,
        "RAG_INDEX_PATH",
        index_path
    )

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

    mock_model.assert_called_once()

    mock_rag.assert_called_once_with(
        question="SSE 的 done 事件表示什么？",
        index_path=index_path,
        embedding_model=fake_model,
        k=2
    )


def test_rag_ask_empty_question(monkeypatch, tmp_path):
    index_path = tmp_path / "rag_index.json"
    index_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        api,
        "RAG_INDEX_PATH",
        index_path
    )

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

    with TestClient(api.app) as client:
        response = client.post(
            "/rag/ask",
            json={
                "question": "    ",
                "k": 2
            }
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "问题不能为空"
    mock_model.assert_not_called()
    mock_rag.assert_not_called()

def test_rag_ask_missing_index(monkeypatch, tmp_path):
    index_path = tmp_path / "not_exist.json"

    monkeypatch.setattr(
        api,
        "RAG_INDEX_PATH",
        index_path
    )

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

    with TestClient(api.app) as client:
        response = client.post(
            "/rag/ask",
            json={
                "question": "SSE 的 done 事件表示什么？",
                "k": 2
            }
        )

    assert response.status_code == 503
    mock_model.assert_not_called()
    mock_rag.assert_not_called()


def test_rag_ask_invalid_k(monkeypatch, tmp_path):
    index_path = tmp_path / "rag_index.json"
    index_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        api,
        "RAG_INDEX_PATH",
        index_path
    )

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