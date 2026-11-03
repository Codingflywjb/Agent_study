import pytest
from practice.rag_index_metadata import validate_index_metadata,create_index_payload


def test_validate_index_metadata_success():
    metadata = {
        "schema_version": 2,
        "embedding_model_name": "BAAI/bge-small-zh-v1.5",
        "embedding_dim": 512
    }

    result = validate_index_metadata(
        metadata,
        expected_model_name="BAAI/bge-small-zh-v1.5",
        expected_dim=512
    )

    assert result is None


def test_validate_index_metadata_invalid_version():
    metadata = {
        "schema_version": 1,
        "embedding_model_name": "BAAI/bge-small-zh-v1.5",
        "embedding_dim": 512
    }
    with pytest.raises(ValueError):
        validate_index_metadata(
            metadata,
            expected_model_name="BAAI/bge-small-zh-v1.5",
            expected_dim=512
        )

def test_validate_index_metadata_model_mismatch():
    metadata = {
        "schema_version": 2,
        "embedding_model_name": "BAAI/bge-small-zh-v1.5",
        "embedding_dim": 512
    }
    with pytest.raises(ValueError):
        validate_index_metadata(
            metadata,
            expected_model_name="another-model",
            expected_dim=512
        )


def test_validate_index_metadata_dim_mismatch():
    metadata = {
        "schema_version": 2,
        "embedding_model_name": "BAAI/bge-small-zh-v1.5",
        "embedding_dim": 512
    }
    with pytest.raises(ValueError):
        validate_index_metadata(
            metadata,
            expected_model_name="BAAI/bge-small-zh-v1.5",
            expected_dim=768
        )

def test_validate_index_metadata_missing_model():
    metadata = {
        "schema_version": 2,
        "embedding_dim": 512
    }
    with pytest.raises(ValueError):
        validate_index_metadata(
            metadata,
            expected_model_name="BAAI/bge-small-zh-v1.5",
            expected_dim=512
        )

def test_create_index_payload_success():
    chunks = [
        {
            "source": "sse.txt",
            "chunk_id": 0,
            "content": "done 事件表示正常结束",
            "embedding": [0.1, 0.2, 0.3]
        }
    ]

    payload = create_index_payload(
        indexed_chunks=chunks,
        model_name="test-model",
        embedding_dim=3
    )

    assert payload["metadata"] == {
        "schema_version": 2,
        "embedding_model_name": "test-model",
        "embedding_dim": 3
    }

    assert payload["chunks"] == chunks


def test_create_index_payload_empty_model():
    chunks = [
        {
            "source": "sse.txt",
            "chunk_id": 0,
            "content": "done 事件表示正常结束",
            "embedding": [0.1, 0.2, 0.3]
        }
    ]

    with pytest.raises(ValueError,match="Invalid model_name"):
        create_index_payload(
            indexed_chunks=chunks,
            model_name="  ",
            embedding_dim=3
        )



def test_create_index_payload_invalid_dim():
    chunks = [
        {
            "source": "sse.txt",
            "chunk_id": 0,
            "content": "done 事件表示正常结束",
            "embedding": [0.1, 0.2, 0.3]
        }
    ]
    with pytest.raises(ValueError,match="Invalid embedding_dim"):
        create_index_payload(
            indexed_chunks=chunks,
            model_name="test-model",
            embedding_dim=0
        )


def test_create_index_payload_chunk_dim_mismatch():
    chunks = [
        {
            "source": "sse.txt",
            "chunk_id": 0,
            "content": "done 事件表示正常结束",
            "embedding": [0.1, 0.2, 0.3]
        }
    ]
    with pytest.raises(ValueError,match="Inconsistent embedding_dim"):
        create_index_payload(
            indexed_chunks=chunks,
            model_name="test-model",
            embedding_dim=512
        )


def test_create_index_payload_empty_chunks():

    payload = create_index_payload(
        indexed_chunks=[],
        model_name="test-model",
        embedding_dim=3
    )
    assert payload["chunks"] == []
    assert payload["metadata"]["embedding_dim"] == 3
    assert payload["metadata"]["schema_version"] == 2
    assert payload["metadata"]["embedding_model_name"] == "test-model"