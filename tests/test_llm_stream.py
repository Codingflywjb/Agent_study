from unittest.mock import  MagicMock

import pytest

from practice.llm_stream import consume_stream

def test_consume_stream_normal(capsys):
    chunk1=MagicMock()
    chunk1.choices[0].delta.content = "你好,"

    chunk2=MagicMock()
    chunk2.choices[0].delta.content = "我是DeepSeek!"

    fake_stream = [chunk1, chunk2]

    answer = consume_stream(fake_stream)

    assert answer == "你好,我是DeepSeek!"

    captured = capsys.readouterr()
    assert captured.out == "你好,我是DeepSeek!\n"

def test_consume_stream_empty_choice():
    chunk=MagicMock()
    chunk.choices = []
    chunk_1=MagicMock()
    chunk_1.choices[0].delta.content = "只因你太美,"
    chunk_2=MagicMock()
    chunk_2.choices[0].delta.content = "ohbaby"
    fake_stream = [chunk,chunk_1, chunk_2]
    answer=consume_stream(fake_stream)
    assert answer == "只因你太美,ohbaby"


def test_consume_stream_empty_content():
    chunk=MagicMock()
    chunk.choices[0].delta.content = "   "
    fake_stream = [chunk]
    with pytest.raises(ValueError,match="没有有效文本"):
        consume_stream(fake_stream)