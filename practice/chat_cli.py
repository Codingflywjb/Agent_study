from practice.llm_client import ask_llm_stream,LLMServiceError

def main() -> None:
    history : list[dict] = []

    print("欢迎使用 DeepSeek 多轮聊天程序")
    print("输入 exit 退出程序")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "exit":
            break
        if user_input == "":
            print("输入不能为空")
            continue
        try:
            response = ask_llm_stream(user_input,history)
            new_messages=[
                {"role":"user","content":user_input},
                {"role":"assistant","content":response}
            ]
            history.extend(new_messages)
        except LLMServiceError as e:
            print(f"用户模型服务调用失败:{e}")
            continue
        except ValueError as e:
            print(f"用户输入或响应数据有误:{e}")
            continue

if __name__ == "__main__":
    main()
