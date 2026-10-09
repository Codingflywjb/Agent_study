
from app.calculator import add,multiply,divide,subtract

def main():

    while True:
        try:
            # 1. 获取两个数字
            x=float(input("输入x数值："))
            y=float(input("输入y数值："))
            # 2. 获取运算符
            operator=input("输入运算符：").strip()

            # 3. 根据运算符调用函数
            if operator == "+":
                result=add(x, y)
            elif operator == "-":
                result=subtract(x, y)
            elif operator == "*":
                result=multiply(x, y)
            elif operator == "/":
                result=divide(x, y)
            else:
                print("运算符错误")
                return
            # 4. 输出结果
            print(result)
            pass

        except ValueError:
            print("输入错误")

        except ZeroDivisionError:
            print("除数不能为0")

        # 5. 询问用户是否继续
        con=input("是否继续计算？（y/n）：")
        if con=="y":
            continue
        elif con=="n":
            break
        # 6. 不继续则 break

if __name__ == "__main__":
    main()