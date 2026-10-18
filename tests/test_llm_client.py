from unittest.mock import MagicMock
import pytest
from practice import llm_client
from openai import APIConnectionError, OpenAI
from httpx2 import Request

def test_ask_llm_success(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-test-key")

    fake_client = MagicMock()

    fake_client.chat.completions.create.return_value.model_dump.return_value = {
        "choices": [
            {
                "message" : {"content" : "  你好，我是deepseek！  "}
            }
        ]
    }

    monkeypatch.setattr(llm_client, "OpenAI", MagicMock(return_value=fake_client))

    result = llm_client.ask_llm("你好")

    assert result ==  "你好，我是deepseek！"
    fake_client.chat.completions.create.assert_called_once()

def test_ask_llm_missing_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError):
        llm_client.ask_llm("你好")

def test_ask_llm_connection_error(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-test-key")
    fake_client = MagicMock()
    fake_error= APIConnectionError(
        request=Request("POST","https://api.deepseek.com")
    )
    fake_client.chat.completions.create.side_effect = fake_error
    monkeypatch.setattr(llm_client, "OpenAI", MagicMock(return_value=fake_client))
    with pytest.raises(llm_client.LLMServiceError,match="网络连接异常") as exc_info:
        llm_client.ask_llm("你好")

    fake_client.chat.completions.create.assert_called_once()

    assert isinstance(exc_info.value.__cause__, APIConnectionError)

def test_ask_llm_with_history(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-test-key")
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value.model_dump.return_value = {
        "choices": [
            {
                "message" : { "content" : "FastAPI很适合开发API"}
            }
        ]
    }
    history = [
        {
            "role":"user",
            "content":"什么是FastAPI?"
        },
        {
            "role":"assistant",
            "content":"FastAPI是一种API"
        }
    ]
    monkeypatch.setattr(llm_client, "OpenAI", MagicMock(return_value=fake_client))
    result = llm_client.ask_llm("它有什么优点？", history)
    sent_messages = (fake_client.chat.completions.create.call_args.kwargs["messages"])

    assert [msg["role"] for msg in sent_messages] == [
        "system",
        "user",
        "assistant",
        "user"
    ]

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[0]["content"] == "什么是FastAPI?"
    assert history[1]["content"] == "FastAPI是一种API"
    assert result == "FastAPI很适合开发API"
    assert sent_messages[-1]["content"] == "它有什么优点？"
    fake_client.chat.completions.create.assert_called_once()

def test_ask_llm_stream_success(monkeypatch,capsys):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-test-key")
    fake_client = MagicMock()

    chunk1=MagicMock()
    chunk1.choices[0].delta.content = "你好,"

    chunk2 = MagicMock()
    chunk2.choices[0].delta.content = "我是DeepSeek!"

    fake_stream = iter([chunk1,chunk2])

    fake_client.chat.completions.create.return_value = fake_stream
    monkeypatch.setattr(llm_client, "OpenAI", MagicMock(return_value=fake_client))
    result=llm_client.ask_llm_stream("你好")
    assert result == "你好,我是DeepSeek!"
    captured = capsys.readouterr()
    assert captured.out == "你好,我是DeepSeek!\n"
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    fake_client.chat.completions.create.assert_called_once()

def test_ask_llm_stream_connection_error(monkeypatch,capsys):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-test-key")
    fake_client = MagicMock()
    chunk=MagicMock()
    chunk.choices[0].delta.content = "部分回答"

    def broken_stream():
        yield chunk
        raise APIConnectionError(
            request=Request("POST","https://api.deepseek.com")
        )
    fake_client.chat.completions.create.return_value = broken_stream()
    monkeypatch.setattr(llm_client, "OpenAI", MagicMock(return_value=fake_client))
    with pytest.raises(llm_client.LLMServiceError) as exc_info:
        llm_client.ask_llm_stream("你好")
    captured = capsys.readouterr()
    assert "部分回答" in captured.out
    assert "网络连接异常" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, APIConnectionError)
    fake_client.chat.completions.create.assert_called_once()