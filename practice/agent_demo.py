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

    user_text = input("请输入任务或问题：")
    if not user_text.strip():
        print("输入不能为空")
        return
    result = run_agent_loop(
        client=client,
        user_text=user_text,
        max_rounds=5,
        dry_run=True
    )
    print(result)

if __name__ == "__main__":
    main()