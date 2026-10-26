from pathlib import Path
from sentence_transformers import SentenceTransformer

from practice.rag_index import load_vector_index
from practice.rag_embedding import embed_text
from practice.rag_retriever import top_k_search

def search_cached_knowledge(
    query: str,
    index_path: str | Path,
    model: SentenceTransformer,
    k: int = 2
) -> list[dict]:
    if not query.strip():
        raise ValueError("Query is required")

    if k<=0:
        raise ValueError("k must be greater than 0")

    indexed_chunks=load_vector_index(index_path)

    if not indexed_chunks:
        return []

    query_embedding = embed_text(text=query, model=model,is_query=True)

    return top_k_search(query_embedding=query_embedding,chunks=indexed_chunks,k=k)