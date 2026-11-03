from pathlib import Path
from sentence_transformers import SentenceTransformer
from practice.rag_cached_search import search_cached_knowledge_v2
from practice.rag_index import (
    build_vector_index,
    save_vector_index_v2,
    load_vector_index_v2,
)

from app.config import load_rag_settings

def main():
    settings = load_rag_settings()
    model_name =settings.embedding_model_name
    docs_dir=Path("knowledge")
    v2_index_path = Path("data/rag_index_v2.json")
    model=SentenceTransformer(model_name)
    print("当前模型名称：",model_name)
    indexed_chunks = build_vector_index(
        directory=docs_dir,
        model=model,
        chunk_size=200,
        overlap=30
    )
    if not indexed_chunks:
        raise ValueError("No chunks generated")
    embedding_dim = len(indexed_chunks[0]["embedding"])
    save_vector_index_v2(
        indexed_chunks=indexed_chunks,
        index_path=v2_index_path,
        model_name=model_name,
        embedding_dim=embedding_dim
    )
    loaded_chunks=load_vector_index_v2(
        index_path=v2_index_path,
        expected_model_name=model_name,
        expected_dim=embedding_dim
    )
    assert len(loaded_chunks) == len(indexed_chunks)
    print("Chunk向量数量：",len(indexed_chunks))
    print("向量维度：",embedding_dim)
    results = search_cached_knowledge_v2(
        query="SSE 的 done 事件表示什么？",
        index_path=v2_index_path,
        model=model,
        expected_model_name=model_name,
        k=2
    )

    for result in results:
        print("来源：", result["source"])
        print("相似度：", result["score"])
        print("内容：", result["content"])
        print("-" * 30)

if __name__ == "__main__":
    main()