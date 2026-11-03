import os
from openai import OpenAI
from dotenv import load_dotenv
from practice.agent_tools import (
    add_task_tool,
    sdk_tool_call_to_dict,
    validate_tool_call,
)
load_dotenv() ##负责把 .env 文件里的配置加载到当前 Python 进程的环境变量中

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    timeout=30.0,
    max_retries=1,
)

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "user",
            "content": "请帮我添加一个任务：明天学习 FastAPI"
        }
    ],
    tools=[add_task_tool],
    tool_choice="required",
    extra_body={"thinking": {"type": "disabled"}},
)

message = response.choices[0].message

tool_calls = message.tool_calls
if not tool_calls:
    print("模型没有发起工具调用")
for tool_call in tool_calls:
    tool_call_dict = sdk_tool_call_to_dict(tool_call)
    tool_name,auguments=validate_tool_call(tool_call_dict)
    print("工具名称：",tool_name,"   工具参数:",auguments)