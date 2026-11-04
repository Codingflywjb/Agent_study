import os
from openai import OpenAI
from dotenv import load_dotenv

from practice.agent_loop import run_agent_loop


def main():
    load_dotenv()

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        timeout=30.0,
        max_retries=1,
    )

    answer = run_agent_loop(
        client=client,
        user_text="请使用 add_task 工具，帮我添加一个任务：明天学习 FastAPI。",
        max_rounds=3
    )

    print("最终回答：", answer)


if __name__ == "__main__":
    main()