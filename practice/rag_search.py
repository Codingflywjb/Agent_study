from pathlib import Path
from sentence_transformers import SentenceTransformer

from practice.rag_pipeline import build_chunks
from practice.rag_embedding import embed_text
from practice.rag_retriever import top_k_search

def search_knowledge(
    query: str,
    directory: str | Path,
    model: SentenceTransformer,
    k: int = 2,
    chunk_size: int = 200,
    overlap: int = 30
) -> list[dict]:
    if not query.strip():
        raise ValueError("query is required")

    if k<=0:
        raise ValueError("k must be greater than 0")

    chunks=build_chunks(directory,chunk_size=chunk_size,overlap=overlap)

    if not chunks:
        return []

    indexed_chunks = []
    for chunk in chunks:
        if not chunk["content"].strip():
            continue

        embedding=embed_text(chunk["content"],model)
        indexed_chunks.append({
            **chunk,
            "embedding": embedding
        })

    query_embedding = embed_text(query,model,is_query=True)
    return top_k_search(query_embedding,indexed_chunks,k=k)


