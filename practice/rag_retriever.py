from practice.rag_similarity import cosine_similarity

def top_k_search(query_embedding:list[float],chunks:list[dict],k:int = 2) -> list[dict]:
    if k<=0:
        raise ValueError("k must be greater than 0")

    if not chunks:
        return []

    scored_chunks = []

    for chunk in chunks:
        score = cosine_similarity(query_embedding,chunk["embedding"])
        scored_chunks.append(
                {
                "source":chunk["source"],
                "chunk_id":chunk["chunk_id"],
                "content":chunk["content"],
                "score":score,
            }
        )

    scored_chunks=sorted(scored_chunks,key=lambda x:x["score"],reverse=True)
    return scored_chunks[:k]