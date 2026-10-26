import pytest
from practice.rag_prompt import build_rag_prompt


def test_build_rag_prompt_success():
    chunks = [
        {
            "source": "fastapi.txt",
            "chunk_id": 0,
            "content": "FastAPI 支持 StreamingResponse",
            "score": 0.92
        }
    ]

    result = build_rag_prompt(
        "FastAPI 如何实现流式输出？",
        chunks
    )

    assert "fastapi.txt" in result
    assert "片段：0" in result
    assert "FastAPI 支持 StreamingResponse" in result
    assert "FastAPI 如何实现流式输出？" in result
    assert "资料不足时明确说明" in result

def test_build_rag_prompt_multiple_chunks():
    chunks = [
        {
            "source": "fastapi.txt",
            "chunk_id": 0,
            "content": "FastAPI 支持 StreamingResponse",
            "score": 0.92
        },
        {
            "source": "sse.txt",
            "chunk_id": 1,
            "content": "SSE 可以持续发送服务器事件",
            "score": 0.85
        }
    ]

    result = build_rag_prompt("FastAPI 如何实现流式输出？",chunks)

    assert "fastapi.txt" in result
    assert "sse.txt" in result
    assert "FastAPI 支持 StreamingResponse" in result
    assert "SSE 可以持续发送服务器事件" in result
    assert result.index("FastAPI 支持 StreamingResponse") < result.index("SSE 可以持续发送服务器事件")
    assert "片段：0" in result
    assert "片段：1" in result

def test_build_rag_prompt_empty_chunks():
    chunks = []
    result = build_rag_prompt("FastAPI 如何实现流式输出？", chunks)
    assert "无可用参考资料" in result
    assert "FastAPI 如何实现流式输出？" in result

def test_build_rag_prompt_invalid_query():
    chunks = [
        {
            "source": "fastapi.txt",
            "chunk_id": 0,
            "content": "FastAPI 支持 StreamingResponse",
            "score": 0.92
        }
    ]
    with pytest.raises(ValueError):
        build_rag_prompt("",chunks)
    with pytest.raises(ValueError):
        build_rag_prompt("    ",chunks)