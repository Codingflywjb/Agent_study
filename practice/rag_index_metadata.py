

def validate_index_metadata(
    metadata: dict,
    expected_model_name: str,
    expected_dim: int | None = None
) -> None:
    schema_version = metadata.get("schema_version")
    if schema_version != 2:
        raise ValueError("Invalid schema version")

    embedding_model_name = metadata.get("embedding_model_name")
    if embedding_model_name != expected_model_name:
        raise ValueError("Invalid embedding_model_name")

    if expected_dim is not None:
        embedding_dim = metadata.get("embedding_dim")
        if embedding_dim != expected_dim:
            raise ValueError("Invalid embedding_dim")

def create_index_payload(
    indexed_chunks: list[dict],
    model_name: str,
    embedding_dim: int
) -> dict:
    if not model_name.strip():
        raise ValueError("Invalid model_name")

    if embedding_dim <= 0:
        raise ValueError("Invalid embedding_dim")

    for chunk in indexed_chunks:
        if len(chunk["embedding"]) != embedding_dim:
            raise ValueError("Inconsistent embedding_dim")

    metadata = {
        "schema_version": 2,
        "embedding_model_name": model_name,
        "embedding_dim": embedding_dim
    }

    return {
        "metadata": metadata,
        "chunks": indexed_chunks
    }