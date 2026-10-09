def add(a:float, b:float) -> float:
    return a+b

## ->int表示返回值类型标注


def multiply(a:float, b:float) -> float:
    return a*b

def subtract(a:float, b:float) -> float:
    return a-b

def divide(a:float, b:float) -> float:
    if b == 0:
        print("Divide by zero")
        return 0
    return a/b