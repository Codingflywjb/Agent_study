def filter_retrieved_chunks(chunks:list[dict],min_score:float = 0.45) -> list[dict]:
    filtered_chunks = []

    for chunk in chunks:
        if chunk["score"] >= min_score:
            filtered_chunks.append(chunk)

    return filtered_chunks