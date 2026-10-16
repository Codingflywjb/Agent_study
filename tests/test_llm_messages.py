import pytest

from practice.llm_messages import extract_llm_content


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