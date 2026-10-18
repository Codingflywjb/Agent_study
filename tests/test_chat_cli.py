
from practice import chat_cli
from practice.llm_client import LLMServiceError


def test_chat_cli_two_turns(monkeypatch):
    inputs = iter([
        "什么是FastAPI？",
        "它有什么优点？",
        "exit"
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": next(inputs)
    )

    snapshots = []
    questions = []

    def fake_ask(question, history):
        # TODO 1：保存本轮调用时 history 的快照
        questions.append(question)
        snapshots.append([msg.copy() for msg in history])
        if question == "什么是FastAPI？":
            return "FastAPI是一个Python Web框架。"

        return "FastAPI支持自动生成API文档。"


    monkeypatch.setattr(chat_cli,"ask_llm_stream",fake_ask)

    chat_cli.main()

    assert questions == [
        "什么是FastAPI？",
        "它有什么优点？"
    ]

    assert snapshots[0] == []
    assert snapshots[1] == [
        {
            "role": "user",
            "content":"什么是FastAPI？"
        },
        {
            "role": "assistant",
            "content":"FastAPI是一个Python Web框架。"
        }
    ]

    assert len(snapshots) == 2

def test_chat_cli_failure_does_not_pollute_history(monkeypatch):
    inputs = iter([
        "第一次提问",
        "第二次提问",
        "exit"
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": next(inputs)
    )

    call_count=0
    snapshots = []
    questions = []
    def fake_ask(question, history):
        nonlocal call_count
        call_count+=1
        questions.append(question)
        snapshots.append([msg.copy() for msg in history])
        if question == "第一次提问":
            raise LLMServiceError("模型连接失败")

        return "模型成功回答"

    monkeypatch.setattr(chat_cli, "ask_llm_stream", fake_ask)
    chat_cli.main()
    assert snapshots[0] == []
    assert snapshots[1] == []

def test_chat_cli_stream_failure(monkeypatch,capsys):
    inputs = iter([
        "什么是LangGraph？",
        "你好",
        "exit"
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": next(inputs)
    )

    snapshots = []
    call_count = 0

    def fake_stream_ask(question, history):
        nonlocal call_count
        call_count += 1

        # TODO 1：保存 history 快照
        snapshots.append([msg.copy() for msg in history])
        if call_count == 1:
            # 模拟已经打印部分回答
            print("部分回答", end="", flush=True)

            # TODO 2：抛出 LLMServiceError
            # 错误消息："模拟流式断网"
            raise LLMServiceError("模拟流式断网")
        # 第二轮模拟正常流式输出
        print("第二轮正常回答")
        return "第二轮正常回答"

    # TODO 3：使用 monkeypatch.setattr
    # 替换 chat_cli.ask_llm_stream
    monkeypatch.setattr(chat_cli, "ask_llm_stream", fake_stream_ask)
    # TODO 4：运行 chat_cli.main()
    chat_cli.main()
    # TODO 5：验证 snapshots == [[], []]
    assert  snapshots == [[],[]]
    # TODO 6：使用 capsys 检查终端输出
    # 包含 "部分回答"
    # 包含 "模拟流式断网"
    # 包含 "第二轮正常回答"
    captured = capsys.readouterr()
    assert "部分回答" in captured.out
    assert "模拟流式断网" in captured.out
    assert "第二轮正常回答" in captured.out
    # TODO 7：验证 call_count == 2
    assert call_count == 2