import json

FILE_PATH = "data/tasks.json"


def save_tasks(tasks):
    # TODO: 使用 with open() 和 json.dump(),设置 ensure_ascii=False 和 indent=4
    with open(FILE_PATH, "w",encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False,indent=4)


def load_tasks():
    # TODO: 使用 with open() 和 json.load()
    try:
        with open(FILE_PATH, "r",encoding="utf-8") as f:
            tasks = json.load(f)
            return tasks

    except FileNotFoundError:
        return []
    # 如果文件不存在，返回 []

def add_task(title):
    # 1. 读取原来的任务列表
    tasks = load_tasks()
    # 2. 为新任务生成 ID
    # 暂时可以使用 len(tasks) + 1
    # 假设目前没有删除任务的操作
    # 3. 创建一个新任务字典
    new_task = {
        "id": len(tasks)+1,
        "title": title,
        "completed": False
    }
    # 4. 将新任务添加到列表
    tasks.append(new_task)
    # 5. 保存更新后的任务列表
    save_tasks(tasks)




if __name__ == "__main__":

    while True:
        op=input("是否添加新任务(y/n):").strip().lower()
        if op=="y":
            title = input("请输入新任务：").strip()

            if title:
                add_task(title)
                print("任务添加成功！")
            else:
                print("任务名称不能为空！")

            print("\n当前任务列表：")

            tasks = load_tasks()

            for task in tasks:
                print(task)
        elif op=="n":
            break
        else:
            print("输入错误，只能输入y/n")