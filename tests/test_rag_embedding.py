import numpy as np
import pytest

from unittest.mock import MagicMock

from practice.rag_embedding import embed_text, QUERY_PREFIX

def test_embed_document():
    model = MagicMock()
    model.encode.return_value = np.array([0.6, 0.8])

    result = embed_text(
        "FastAPI 支持流式输出",
        model
    )

    assert result == [0.6, 0.8]

    model.encode.assert_called_once_with(
        "FastAPI 支持流式输出",
        normalize_embeddings=True
    )


def test_embed_query():
    model = MagicMock()
    model.encode.return_value = np.array([0.6, 0.8])
    question="FastAPI 支持流式输出"
    # TODO 1：
    # 调用 embed_text()
    # is_query=True
    result = embed_text(
        question,
        model,
        is_query=True
    )

    # TODO 2：
    # 使用 assert_called_once_with()
    # 验证传给 encode() 的文本是：
    # QUERY_PREFIX + 原始问题
    # 并验证 normalize_embeddings=True

    model.encode.assert_called_once_with(QUERY_PREFIX+question,normalize_embeddings=True)

def test_embed_empty_text():
    model = MagicMock()

    # TODO 3：
    # 使用 pytest.raises(ValueError)
    # 验证空字符串 ""
    with pytest.raises(ValueError):
        embed_text("", model)

    # TODO 4：
    # 使用 pytest.raises(ValueError)
    # 验证纯空格字符串 "   "
    with pytest.raises(ValueError):
        embed_text("     ", model)

    # TODO 5（进阶）：
    # 验证非法输入时
    # model.encode() 从未被调用
    model.encode.assert_not_called()
