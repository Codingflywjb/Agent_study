import pytest
from practice.rag_loader import load_documents

def test_load_documents_success(tmp_path):   #注意是tmp_path 而不是temp_path
    file1=tmp_path / 'a.txt' #创建文件a
    file1.write_text("Python基础",encoding='utf-8')#a中写入“Python基础”
    file2=tmp_path / 'b.txt'
    file2.write_text("FastAPI开发",encoding='utf-8')

    documents=load_documents(tmp_path) #获取目录下所有txt文本信息

    assert len(documents) == 2 #获得两个有效文本

    content_1=documents[0]
    content_2=documents[1]

    assert content_1["source"] == "a.txt"
    assert content_1["content"] == "Python基础"

    assert content_2["source"] == "b.txt"
    assert content_2["content"] == "FastAPI开发"

def test_load_documents_skip_empty(tmp_path):
    file_empty=tmp_path / 'empty.txt'
    file_empty.write_text("   ",encoding='utf-8')

    file_valid=tmp_path / 'valid.txt'
    file_valid.write_text("RAG 检索增强生成",encoding='utf-8')

    documents=load_documents(tmp_path)
    assert len(documents) == 1
    valid_content=documents[0]
    assert valid_content["source"] == "valid.txt"
    assert valid_content["content"] == "RAG 检索增强生成"

def test_load_documents_missing_dir(tmp_path):
    path=tmp_path / "not_exist" #创建一个空目录
    with pytest.raises(FileNotFoundError):
        load_documents(path)
