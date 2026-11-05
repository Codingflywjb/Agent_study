from practice.agent_tools import (
    add_task_tool,
    sdk_tool_call_to_dict,
    validate_tool_call
)
from practice.agent_tool_messages import build_tool_result_message
from practice.agent_tool_errors import safe_simulate_tool_call

def decide_next_action(assistant_message)->str:
    tool_calls = assistant_message.tool_calls or []
    if not tool_calls:
        return "final"
    return "tool"

def run_agent_loop(
        client,
        user_text: str,
        max_rounds: int = 5
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "你是任务管理助手。"
                "工具目前处于模拟模式，任务不会实际保存。"
                "必须根据工具执行结果准确回答用户，"
                "不得声称模拟任务已成功保存。"
            )
        },
        {
            "role": "user",
            "content": user_text
        }
    ]
    rounds = 0
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")
    while rounds < max_rounds:
        rounds += 1
        response=client.chat.completions.create(
            model="deepseek-flash",
            tool_choice="auto",
            messages=messages.copy(),
            tools=[add_task_tool],
            extra_body={"thinking":{"type":"disabled"}}
        )
        assistant_message=response.choices[0].message ##
        result=decide_next_action(assistant_message)
        print(
            f"[第 {rounds} 轮] "
            f"action={result}, "
            f"tool_calls={len(assistant_message.tool_calls or [])}"
        )
        if result == "final":
            return assistant_message.content
        if rounds == max_rounds:
            break
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            tool_call_dict = sdk_tool_call_to_dict(tool_call)
            result=safe_simulate_tool_call(tool_call_dict)
            tool_message=build_tool_result_message(tool_call_id=tool_call.id,result=result)
            messages.append(tool_message)
    return "已达到最大请求轮数，任务执行状态未确认。"