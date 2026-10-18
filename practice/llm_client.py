
import os

from openai import (OpenAI,APITimeoutError,APIConnectionError,AuthenticationError,RateLimitError,APIStatusError,)

from app.config import MODEL_NAME
from practice.llm_messages import (
    build_chat_messages,
    extract_llm_content,
)


class LLMServiceError(RuntimeError):
    """大模型服务调用失败。"""


def ask_llm(user_input: str,history:list[dict] | None = None) -> str:
    if history is None:
        history = []
    messages = build_chat_messages(history,user_input)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("无法找到api_key")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=30.0,
        max_retries=0,
    )

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
        raise LLMServiceError("Deepseek请求超时") from e
    except AuthenticationError as e:
        raise LLMServiceError("Deepseek身份认证失败") from e
    except RateLimitError as e:
        raise LLMServiceError("Deepseek请求过于频繁") from e
    except APIConnectionError as e:
        raise LLMServiceError("Deepseek网络连接异常") from e
    except APIStatusError as e:
        raise LLMServiceError(f"Deepseek其他HTTP请求失败，状态码：{e.status_code}") from e

    response_dict = response.model_dump()
    content = extract_llm_content(response_dict)

    return content



if __name__ == "__main__":
    question = input("请输入你的问题：")
    answer = ask_llm(question)
    print("DeepSeek：", answer)
