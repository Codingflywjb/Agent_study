import os
from openai import OpenAI
from dotenv import load_dotenv
from unittest.mock import patch
import practice.agent_loop as agent_loop
from practice.agent_loop import run_agent_loop


def main():
    load_dotenv()

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        timeout=30.0,
        max_retries=1,
    )


    original_safe = agent_loop.safe_simulate_tool_call  ##保存原始函数引用，避免临时替换后再次调用被替换的函数而导致递归。
    attempts = {"count": 0}

    def simulate_one_failure(tool_call):
        attempts["count"] += 1

        if attempts["count"] == 1:
            print("[故障注入] 第一次工具调用模拟失败")

            return {
                "status": "error",
                "error": "simulated_tool_failure",
                "saved": False
            }

        print("[故障注入] 后续工具调用正常模拟")
        return original_safe(tool_call)

    with patch.object(  ##临时替换 agent_loop 模块中的 safe_simulate_tool_call()。离开 with 后，原函数会自动恢复。
        agent_loop,
        "safe_simulate_tool_call",
        side_effect=simulate_one_failure
    ):
        answer = agent_loop.run_agent_loop(
            client=client,
            user_text="请使用 add_task 工具添加任务：明天学习 FastAPI",
            max_rounds=4
        )

    print("最终回答：", answer)
    print("工具处理次数：", attempts["count"]) ##记录工具处理次数。第一次返回模拟错误，之后再调用原来的安全模拟函数。



if __name__ == "__main__":
    main()