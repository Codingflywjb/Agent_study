import pytest
from practice.rag_splitter import split_text

def test_split_text_with_overlap():
    result = split_text(
        "ABCDEFGHIJ",
        chunk_size=4,
        overlap=1
    )

    assert result == ["ABCD", "DEFG", "GHIJ"]


def test_split_text_without_overlap():
    # TODO 1：
    # 输入 "ABCDE"
    # chunk_size=3, overlap=0
    # 验证结果为 ["ABC", "DE"]

    result = split_text(
        "ABCDE",
        chunk_size=3,
        overlap=0
    )
    assert result == ["ABC", "DE"]



def test_split_text_short_text():
    # TODO 2：
    # 输入 "ABC"
    # chunk_size=5, overlap=1
    # 验证短文本不会被重复切分
    result = split_text(
        "ABC",
        chunk_size=5,
        overlap=1
    )
    assert result == ["ABC"]


def test_split_text_empty():
    # TODO 3：
    # 验证空字符串返回 []
    # 额外验证纯空格字符串不会丢失
    result = split_text(
        "",
        chunk_size=3,
        overlap=0
    )
    assert result == []

    spaces_result = split_text(
        "   ",
        chunk_size=2,
        overlap=0
    )

    assert spaces_result == ["  ", " "]



def test_split_text_invalid_params():
    # TODO 4：
    # 使用 pytest.raises(ValueError)
    # 分别验证：
    # chunk_size=0
    # overlap=-1
    # overlap=chunk_size

    with pytest.raises(ValueError):
        split_text(
            "ABCDE",
            chunk_size=0,
            overlap=0
        )

    with pytest.raises(ValueError):
        split_text(
            "ABCDE",
            chunk_size=3,
            overlap=-1
        )

    with pytest.raises(ValueError):
        split_text(
            "ABCDE",
            chunk_size=3,
            overlap=3
        )