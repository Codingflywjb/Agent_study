from unittest.mock import MagicMock
import pytest
from practice import llm_client
from practice.llm_client import LLMTimeoutError, LLMConnectionError,ask_llm
from openai import APIConnectionError, OpenAI, APITimeoutError
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

def test_iter_llm_chunks_success(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-test-key")
    fake_client = MagicMock()

    chunk1 = MagicMock()
    chunk1.choices[0].delta.content = "你好，"

    chunk2 = MagicMock()
    chunk2.choices[0].delta.content = "我是DeepSeek！"

    empty_chunk = MagicMock()
    empty_chunk.choices = []

    fake_stream = iter([chunk1, empty_chunk, chunk2])

    # TODO 1：
    # 设置 fake_client.chat.completions.create.return_value
    # 返回 fake_stream
    fake_client.chat.completions.create.return_value = fake_stream

    # TODO 2：
    # monkeypatch 替换 llm_client.OpenAI
    monkeypatch.setattr(llm_client, "OpenAI", MagicMock(return_value=fake_client))
    # TODO 3：
    # 调用 iter_llm_chunks("你好")
    # 注意：此时只是获得生成器
    generator=llm_client.iter_llm_chunks("你好") #创建生成器

    # 尚未遍历，所以 API 方法没有被调用
    fake_client.chat.completions.create.assert_not_called()
    # TODO 4：
    # 使用 list() 完整消费生成器
    results = list(generator) #执行生成器函数
    # 遍历之后，API 方法被调用一次
    fake_client.chat.completions.create.assert_called_once()
    # TODO 5：
    # 断言获得的结果：
    # ["你好，", "我是DeepSeek！"]
    assert results == ["你好，", "我是DeepSeek！"]
    # TODO 6：
    # 验证 create() 的 stream 参数为 True
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True

def test_iter_llm_chunks_connection_error(monkeypatch):
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
    generator=llm_client.iter_llm_chunks("你好")
    first_chunk = next(generator)
    assert first_chunk == "部分回答"

    with pytest.raises(llm_client.LLMServiceError) as exc_info:
       next(generator)
    assert "网络连接异常" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, APIConnectionError)

def test_iter_llm_chunks_empty_stream(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-test-key")

    fake_client = MagicMock()
    fake_stream = iter([])
    fake_client.chat.completions.create.return_value = fake_stream

    monkeypatch.setattr(llm_client, "OpenAI", MagicMock(return_value=fake_client))

    with pytest.raises(llm_client.LLMServiceError,match="模型未返回有效内容"):
        list(llm_client.iter_llm_chunks("你好"))

def test_iter_llm_chunks_skips_none_content(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-test-key")
    fake_client = MagicMock()
    chunk1 = MagicMock()
    chunk1.choices[0].delta.content = None
    chunk2 = MagicMock()
    chunk2.choices[0].delta.content = "你好"
    fake_stream = iter([chunk1, chunk2])
    fake_client.chat.completions.create.return_value = fake_stream
    monkeypatch.setattr(llm_client, "OpenAI", MagicMock(return_value=fake_client))
    generator=llm_client.iter_llm_chunks("你好")
    results = list(generator)
    assert results == ["你好"]

def test_iter_llm_chunks_preserves_spaces(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-test-key")
    fake_client = MagicMock()

    chunk1 = MagicMock()
    chunk1.choices[0].delta.content = "Hello"
    chunk2 = MagicMock()
    chunk2.choices[0].delta.content = " "
    chunk3 = MagicMock()
    chunk3.choices[0].delta.content = "World"
    fake_stream = iter([chunk1, chunk2, chunk3])
    fake_client.chat.completions.create.return_value = fake_stream
    monkeypatch.setattr(llm_client, "OpenAI", MagicMock(return_value=fake_client))
    generator=llm_client.iter_llm_chunks("Hello")
    results = list(generator)
    assert results == ["Hello", " ", "World"]
    assert "".join(results) == "Hello World"

def test_ask_llm_timeout_conversion(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = (
        APITimeoutError(
            request=MagicMock()
        )
    )
    monkeypatch.setattr(llm_client,"_create_deepseek_client",MagicMock(return_value=mock_client))
    with pytest.raises(LLMTimeoutError) as exc_info:
        ask_llm("你好")

    assert isinstance(
        exc_info.value.__cause__,
        APITimeoutError
    )
    mock_client.chat.completions.create.assert_called_once()

def test_ask_llm_connection_conversion(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = (
        APIConnectionError(
            request=MagicMock()
        )
    )
    monkeypatch.setattr(llm_client,"_create_deepseek_client",MagicMock(return_value=mock_client))
    with pytest.raises(LLMConnectionError) as exc_info:
        ask_llm("你好")

    assert isinstance(
        exc_info.value.__cause__,
        APIConnectionError
    )
    mock_client.chat.completions.create.assert_called_once()

