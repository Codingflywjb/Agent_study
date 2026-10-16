def build_messages(user_input:str) -> list[dict]:
    user_input = user_input.strip()
    if not user_input:
        raise ValueError("用户输入不能为空")
    messages=[
        {
            "role":"system",
            "content":"你是一名专业的Python编程助手"
        },
        {
            "role":"user",
            "content":user_input
        }
    ]
    return messages

def extract_llm_content(response:dict) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError,IndexError,TypeError) as e:
        raise ValueError("LLM响应格式错误") from e
    if not isinstance(content,str):
        raise ValueError("content格式错误")
    content = content.strip()
    if not content:
        raise ValueError("没有有效文本")
    return content


