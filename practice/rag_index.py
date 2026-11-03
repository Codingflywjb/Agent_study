
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer

from practice.rag_index_metadata import create_index_payload, validate_index_metadata
from practice.rag_pipeline import build_chunks
from practice.rag_embedding import embed_text


def build_vector_index(
    directory: str | Path,
    model: SentenceTransformer,
    chunk_size: int = 200,
    overlap: int = 30
) -> list[dict]:

    # TODO 1：
    # 调用 build_chunks()
    # 传递 directory、chunk_size、overlap
    chunks = build_chunks(directory=directory,chunk_size=chunk_size, overlap=overlap)

    indexed_chunks = []

    # TODO 2：
    # 遍历每个 Chunk
    for chunk in chunks:
        # TODO 3：
        # 跳过仅包含空格的 Chunk
        if not chunk["content"].strip():
            continue

        # TODO 4：
        # 使用 embed_text() 生成文档向量
        # 注意：文档不需要 is_query=True
        embedding = embed_text(text=chunk["content"], model=model,is_query=False)

        indexed_chunks.append(
            {
                **chunk,
                "embedding": embedding
            }
        )
        # TODO 5：
        # 构造新字典：
        # {
        #     **chunk,
        #     "embedding": embedding
        # }

    return indexed_chunks

def save_vector_index(
    indexed_chunks: list[dict],
    index_path: str | Path
) -> None:
    path = Path(index_path)
    path.parent.mkdir(parents=True,exist_ok=True)

    with open(path,"w",encoding='utf-8') as f:
        json.dump(indexed_chunks,f,ensure_ascii=False,indent=2)

def load_vector_index(index_path: str | Path) -> list[dict]:
    path = Path(index_path)
    with open(path,"r",encoding='utf-8') as f:
        indexed_chunks = json.load(f)
    return indexed_chunks

def save_vector_index_v2(
    indexed_chunks: list[dict],
    index_path: str | Path,
    model_name: str,
    embedding_dim: int
) -> None:
    payload = create_index_payload(
        indexed_chunks=indexed_chunks,
        model_name=model_name,
        embedding_dim=embedding_dim
    )
    path = Path(index_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding='utf-8') as f:
        json.dump(payload,f,ensure_ascii=False,indent=2)

def load_vector_index_v2(
    index_path: str | Path,
    expected_model_name: str,
    expected_dim: int | None = None
) -> list[dict]:
    path = Path(index_path)
    with open(path,"r",encoding='utf-8') as f:
        payload = json.load(f)
    if not isinstance(payload,dict):
        raise ValueError("Unsupported index format")
    metadata = payload.get("metadata")
    chunks = payload.get("chunks")
    if not isinstance(metadata,dict):
        raise ValueError("Unsupported index metadata format")
    if not isinstance(chunks,list):
        raise ValueError("Unsupported index chunks format")
    validate_index_metadata(
        metadata=metadata,
        expected_model_name=expected_model_name,
        expected_dim=expected_dim
    )

    return chunks

