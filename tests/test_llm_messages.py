import pytest
from practice.llm_messages import extract_llm_content,build_chat_messages


def test_extract_llm_content_normal():
    response = {
        "choices": [
            {
                "message": {
                    "content": " 你好，Python！ "
                }
            }
        ]
    }

    result = extract_llm_content(response)

    assert result == "你好，Python！"

def test_extract_llm_content_empty():
    response = {
        "choices": [
            {
                "message": {
                    "content": ""
                }
            }
        ]
    }
    with pytest.raises(ValueError):
        extract_llm_content(response)

def test_extract_llm_cohoices_empty():
    response = {}
    with pytest.raises(ValueError):
        extract_llm_content(response)

def test_build_chat_messages_normal():
    history = [
        {"role": "user", "content": "什么是Python？"},
        {"role": "assistant", "content": "Python是一种编程语言。"}
    ]

    result = build_chat_messages(history, "  它有什么优点？  ")

    assert len(result) == 4
    assert result[0]["role"] == "system"
    assert result[1] == history[0]
    assert result[2] == history[1]
    assert result[3]["role"] == "user"
    assert result[3]["content"] == "它有什么优点？"

def test_original_history():
    original_history = [
        {"role":"user","content":"你好"}
    ]
    result = build_chat_messages(original_history, "你好吗？")
    assert len(original_history) == 1
    assert original_history[0]["role"] == "user"
    assert original_history[0]["content"] == "你好"

def test_build_chat_messages_abnormal():
    with pytest.raises(ValueError):
        build_chat_messages([],"   ")