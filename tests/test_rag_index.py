import json
import numpy as np
from unittest.mock import MagicMock

import pytest

from practice.rag_index import (build_vector_index,save_vector_index,load_vector_index,save_vector_index_v2,load_vector_index_v2)

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

def test_save_vector_index_v2(tmp_path):
    index_path = tmp_path / "data" / "index.json"

    chunks = [
        {
            "source": "sse.txt",
            "chunk_id": 0,
            "content": "done 事件表示正常结束",
            "embedding": [0.1, 0.2, 0.3]
        }
    ]

    save_vector_index_v2(
        indexed_chunks=chunks,
        index_path=index_path,
        model_name="test-model",
        embedding_dim=3
    )

    assert index_path.is_file()

    data = json.loads(
        index_path.read_text(encoding="utf-8")
    )
    raw_text = index_path.read_text(encoding="utf-8")

    assert "正常结束" in raw_text
    assert data["metadata"]["schema_version"] == 2
    assert data["metadata"]["embedding_model_name"] == "test-model"
    assert data["metadata"]["embedding_dim"] == 3
    assert data["chunks"] == chunks


def test_save_vector_index_v2_invalid_dim(tmp_path):
    index_path = tmp_path / "data" / "index.json"

    chunks = [
        {
            "source": "sse.txt",
            "chunk_id": 0,
            "content": "正常结束",
            "embedding": [0.1, 0.2, 0.3]
        }
    ]


    with pytest.raises(ValueError,match="Inconsistent embedding_dim"):
        save_vector_index_v2(
            indexed_chunks=chunks,
            index_path=index_path,
            model_name="test-model",
            embedding_dim=512
        )
    assert not index_path.exists()
    assert not index_path.parent.exists()

def test_load_vector_index_v2_success(tmp_path):
    index_path = tmp_path / "index.json"

    chunks = [
        {
            "source": "sse.txt",
            "chunk_id": 0,
            "content": "正常结束",
            "embedding": [0.1, 0.2, 0.3]
        }
    ]

    save_vector_index_v2(
        indexed_chunks=chunks,
        index_path=index_path,
        model_name="test-model",
        embedding_dim=3
    )

    result = load_vector_index_v2(
        index_path=index_path,
        expected_model_name="test-model",
        expected_dim=3
    )

    assert result == chunks


def test_load_vector_index_v2_model_mismatch(tmp_path):
    index_path = tmp_path / "index.json"

    chunks = [
        {
            "source": "sse.txt",
            "chunk_id": 0,
            "content": "正常结束",
            "embedding": [0.1, 0.2, 0.3]
        }
    ]

    save_vector_index_v2(
        indexed_chunks=chunks,
        index_path=index_path,
        model_name="model-A",
        embedding_dim=3
    )
    with pytest.raises(ValueError,match="Invalid embedding_model_name"):
        load_vector_index_v2(
            index_path=index_path,
            expected_model_name="model-B",
            expected_dim=3
        )



def test_load_vector_index_v2_old_format(tmp_path):
    index_path = tmp_path / "index.json"
    old_index = [
        {
            "source": "sse.txt",
            "chunk_id": 0,
            "content": "正常结束",
            "embedding": [0.1, 0.2, 0.3]
        }
    ]
    with open(index_path,"w",encoding="utf-8") as f:
        json.dump(
            old_index,f
        )

    with pytest.raises(ValueError,match="Unsupported index format"):
        load_vector_index_v2(
            index_path=index_path,
            expected_model_name="test-model",
            expected_dim=3
        )

def test_load_vector_index_v2_missing_metadata(tmp_path):

    index_path = tmp_path / "index.json"
    with open(index_path,"w",encoding="utf-8") as f:
        json.dump(
            {
                "chunks": []
            },f
        )

    with pytest.raises(ValueError,match="Unsupported index metadata format"):
        load_vector_index_v2(
            index_path=index_path,
            expected_model_name="test-model",
            expected_dim=3
        )



