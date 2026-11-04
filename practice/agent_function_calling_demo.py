import os
from openai import OpenAI
from dotenv import load_dotenv
from practice.agent_tool_messages import build_tool_result_message
from practice.agent_tools import (
    add_task_tool,
    sdk_tool_call_to_dict,
    validate_tool_call,
)

def run_tool_call_demo(client,user_text:str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个任务管理助手。"
                "你必须根据工具的实际执行结果回答用户。"
                "如果工具返回 saved=false 或 status=simulated，"
                "必须明确说明任务尚未保存，"
                "不得声称任务已添加成功。"
            )
        },
        {
            "role": "user",
            "content": user_text
        }
    ]
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages.copy(),
        tools=[add_task_tool],
        tool_choice="required",
        extra_body={"thinking": {"type": "disabled"}},
    )
    assistant_message = response.choices[0].message
    tool_calls = assistant_message.tool_calls or []
    if len(tool_calls) != 1:
        raise ValueError("Expected exactly one tool call")
    tool_call = tool_calls[0]
    tool_call_dict = sdk_tool_call_to_dict(tool_call)
    tool_name, arguments = validate_tool_call(tool_call_dict)
    print("工具名称：", tool_name, "   工具参数:", arguments)
    fake_result = {
        "status": "simulated",  ##不写success是因为我们没有执行真正的 add_task()，不能让模型误以为任务已经保存。
        "title": arguments["title"],
        "saved": False
    }
    messages.append(assistant_message)
    tool_message = build_tool_result_message(tool_call_id=tool_call.id, result=fake_result)
    messages.append(tool_message)
    response_two = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages.copy(),
        tools=[add_task_tool],
        tool_choice="none",  ##希望模型根据工具结果生成最终文字回答，而不是继续调用工具
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response_two.choices[0].message.content

def main():
    load_dotenv()  ##负责把 .env 文件里的配置加载到当前 Python 进程的环境变量中

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        timeout=30.0,
        max_retries=1,
    )
    answer = run_tool_call_demo(client,user_text="请帮我添加一个任务：明天学习 FastAPI")
    print("最终回答：",answer)

if __name__ == "__main__":
    main()
