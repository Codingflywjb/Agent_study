import json
import numpy as np
from unittest.mock import MagicMock

import pytest

from practice.rag_index import (build_vector_index,save_vector_index,load_vector_index)

def test_build_vector_index(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("Hello     World",encoding="utf-8")

    model = MagicMock()
    model.encode.return_value = np.array([0.6,0.8])

    vector_index=build_vector_index(directory=tmp_path,model=model,chunk_size=5,overlap=0)

    assert len(vector_index)==2

    assert vector_index[0]["content"] == "Hello"
    assert vector_index[1]["content"] == "World"

    assert vector_index[0]["chunk_id"] == 0
    assert vector_index[1]["chunk_id"] == 2

    assert model.encode.call_count == 2

def test_save_vector_index(tmp_path):
    indexed_chunks = [
        {
            "source": "fastapi.txt",
            "chunk_id": 0,
            "content": "FastAPI 支持流式输出",
            "embedding": [0.6, 0.8]
        }
    ]

    index_path = tmp_path / "data" / "index.json"
    save_vector_index(indexed_chunks,index_path)

    assert index_path.exists()
    assert "流式输出" in index_path.read_text(
        encoding="utf-8"
    )

    with open(index_path,"r",encoding="utf-8") as f:
        result = json.load(f)
    assert result == indexed_chunks

def test_load_vector_index(tmp_path):
    indexed_chunks = [
        {
            "source": "fastapi.txt",
            "chunk_id": 0,
            "content": "FastAPI 支持流式输出",
            "embedding": [0.6, 0.8]
        }
    ]

    index_path = tmp_path / "index.json"

    # TODO 1：调用 save_vector_index()
    save_vector_index(indexed_chunks,index_path)
    # TODO 2：调用 load_vector_index()
    result=load_vector_index(index_path)
    # TODO 3：验证读取结果与原始数据一致
    assert result == indexed_chunks

def test_load_vector_index_missing_file():
    with pytest.raises(FileNotFoundError):
        load_vector_index("data/not_exist.json")