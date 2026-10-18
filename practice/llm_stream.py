
def consume_stream(stream) -> str:
    parts: list[str] = []

    for chunk in stream:
        # TODO 1：如果 chunk.choices 为空，跳过
        if not chunk.choices:
            continue
        # TODO 2：获取 chunk.choices[0].delta.content
        content = chunk.choices[0].delta.content
        # TODO 3：如果 content 有文本：
        #   立即打印，不换行，并刷新缓冲区
        #   将 content 添加到 parts
        if content:
            print(content,end="",flush=True)
            parts.append(content)

    print()

    # TODO 4：使用 join() 拼接完整文本
    answer = "".join(parts)
    # TODO 5：如果最终没有有效文本，抛出 ValueError
    if not answer.strip():
        raise ValueError("没有有效文本")
    # TODO 6：返回完整文本
    return answer