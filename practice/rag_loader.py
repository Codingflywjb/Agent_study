from pathlib import Path

def load_documents(directory:str | Path) -> list[dict[str, str]]:  #知识库加载器

    folder = Path(directory) #找到diretory目录
    if not folder.is_dir(): #如果目录下没有文件则报错
        raise FileNotFoundError(f'{directory}知识库目录不存在或不是文件夹')
    documents = []
    for file in sorted(folder.glob('*.txt')):#排序目录下的所有txt文件，遍历
        content=file.read_text(encoding='utf-8')#读取txt文件内容

        content=content.strip()#去除首尾空字符
        if not content: #过滤空文件
            continue
        documents.append({
            "source": file.name,
            "content": content
        }) #添加结果内容
    return documents