
import os

from openai import (OpenAI,APITimeoutError,APIConnectionError,AuthenticationError,RateLimitError,APIStatusError,)
from practice.llm_stream import consume_stream
from app.config import MODEL_NAME
from practice.llm_messages import (
    build_chat_messages,
    extract_llm_content,
)
from _collections_abc import Iterator

class LLMServiceError(RuntimeError):
    """大模型服务调用失败。"""

class LLMTimeoutError(LLMServiceError):
    pass

class LLMConnectionError(LLMServiceError):
    pass

class LLMConfigurationError(ValueError):
    """LLM 服务配置不正确。"""

def _create_deepseek_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise LLMConfigurationError("未配置 DEEPSEEK_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=30.0,
        max_retries=0,
    )
    return client

def ask_llm(user_input: str,history:list[dict] | None = None) -> str:
    if history is None:
        history = []
    messages = build_chat_messages(history,user_input)

    client=_create_deepseek_client()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            extra_body={
                "thinking": {"type": "disabled"}
            },
            stream=False
        )
    except APITimeoutError as e:
        raise LLMTimeoutError("Deepseek请求超时") from e
    except AuthenticationError as e:
        raise LLMServiceError("Deepseek身份认证失败") from e
    except RateLimitError as e:
        raise LLMServiceError("Deepseek请求过于频繁") from e
    except APIConnectionError as e:
        raise LLMConnectionError("Deepseek网络连接异常") from e
    except APIStatusError as e:
        raise LLMServiceError(f"Deepseek其他HTTP请求失败，状态码：{e.status_code}") from e

    response_dict = response.model_dump()
    content = extract_llm_content(response_dict)

    return content


def ask_llm_stream(user_input: str,history: list[dict] | None = None) -> str:
    if history is None:
        history = []
    messages = build_chat_messages(history,user_input)

    client = _create_deepseek_client()
    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            extra_body={
                "thinking": {"type": "disabled"}
            },
            stream=True
        )
        # TODO 6：把流式响应交给 consume_stream()
        # 返回最终拼接好的字符串
        answer = consume_stream(response)
        return answer

    # TODO 7：保留原有的 SDK 异常处理分支
    # 包括认证、限流、超时、网络和 HTTP 状态错误
    except APITimeoutError as e:
        raise LLMServiceError("Deepseek请求超时") from e
    except AuthenticationError as e:
        raise LLMServiceError("Deepseek身份认证失败") from e
    except RateLimitError as e:
        raise LLMServiceError("Deepseek请求过于频繁") from e
    except APIConnectionError as e:
        raise LLMServiceError("Deepseek网络连接异常") from e
    except APIStatusError as e:
        raise LLMServiceError(f"Deepseek其他HTTP请求失败，状态码：{e.status_code}") from e

def iter_llm_chunks(user_input: str,history: list[dict] | None = None) -> Iterator[str]: #Iterator[str]表示这个函数返回迭代器，能逐个产生字符串
    if history is None:
        history = []
    messages = build_chat_messages(history,user_input)
    client = _create_deepseek_client()
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            extra_body={
                "thinking": {"type": "disabled"}
            },
            stream=True
        )
        has_valid_content = False

        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if not content:
                continue
            if content.strip():
                has_valid_content = True
            yield content

        if not has_valid_content:
            raise LLMServiceError("模型未返回有效内容")
    except APITimeoutError as e:
        raise LLMServiceError("Deepseek请求超时") from e
    except AuthenticationError as e:
        raise LLMServiceError("Deepseek身份认证失败") from e
    except RateLimitError as e:
        raise LLMServiceError("Deepseek请求过于频繁") from e
    except APIConnectionError as e:
        raise LLMServiceError("Deepseek网络连接异常") from e
    except APIStatusError as e:
        raise LLMServiceError(f"Deepseek其他HTTP请求失败，状态码：{e.status_code}") from e

if __name__ == "__main__":
    question = input("请输入你的问题：")
    answer = ask_llm(question)
    print("DeepSeek：", answer)
