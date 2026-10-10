import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )

from app.task_manager import get_all_tasks, add_task,complete_task,delete_task,validate_task_id

def main():
    while True:
        print("\n======任务管理系统======")
        print("1.添加任务")
        print("2.查看任务")
        print("3.完成任务")
        print("4.删除任务")
        print("0.退出程序")

        choice=input("请选择操作：").strip()

        if choice=="1":
            title=input("请输入要添加的任务:").strip()
            try:
                add_task(title)
                print("添加成功")
            except ValueError as e:
                print("添加失败：",e)
                continue
        elif choice=="2":
            tasks=get_all_tasks()
            for task in tasks:
                print(task)
        elif choice=="3":
            try:
                task_id=int(input("请输入所完成任务的id:").strip())
                validate_task_id(task_id)
            except ValueError:
                print("id输入不合法")
                continue
            success=complete_task(task_id)
            if success:
                print("任务已完成")
            else:
                print("未找到任务")
        elif choice=="4":
            try:
                task_id =int(input("请输入所完成任务的id:").strip())
                validate_task_id(task_id)
            except ValueError:
                print("id输入不合法")
                continue
            success=delete_task(task_id)
            if success:
                print("删除成功")
            else:
                print("删除失败")
        elif choice=="0":
            break
        else:
            print("输入错误，请输入1、2、3、4、0")

if __name__=="__main__":
    main()
