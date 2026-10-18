
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


    monkeypatch.setattr(chat_cli,"ask_llm",fake_ask)

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

    monkeypatch.setattr(chat_cli, "ask_llm", fake_ask)
    chat_cli.main()
    assert snapshots[0] == []
    assert snapshots[1] == []