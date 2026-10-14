
import requests

def get_todo(todo_id:int) -> dict:
    url = f"https://jsonplaceholder.typicode.com/todos/{todo_id}"

    response = requests.get(url,timeout=10)
    response.raise_for_status()
    data = response.json()
    return data

def create_post(title:str,body:str,userId:int) -> dict:
    url = "https://jsonplaceholder.typicode.com/posts"
    payload = {
        "title":title,
        "body":body,
        "userId":userId
    }
    response = requests.post(url,json=payload,timeout=10)
    response.raise_for_status()
    data = response.json()
    return data

def todo_request(todo_id:int) -> None:
    try:
        todo=get_todo(todo_id)
        print(todo['id'])
        print(todo['title'])
        print(todo['completed'])
    except requests.exceptions.RequestException as e:
        print(f"请求失败:{e}")

def submit_post(title:str,body:str,userId:int) -> None:
    try:
        post=create_post(title=title,body=body,userId=userId)
        print(post['id'])
        print(post['title'])
        print(post['body'])
        print(post)
    except requests.exceptions.RequestException as e:
        print(f"请求失败:{e}")




if __name__=="__main__":
    title="学习Agent"
    body="今天学习了HTTP_POST请求"
    userId=1
    submit_post(title=title,body=body,userId=userId)