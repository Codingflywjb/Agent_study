from sentence_transformers import SentenceTransformer

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

def embed_text(
    text: str,
    model: SentenceTransformer,
    is_query: bool = False
) -> list[float]:

    # TODO 1：
    # 如果文本为空或仅包含空格
    # 抛出 ValueError
    if not text.strip():
        raise ValueError("text is required")

    # TODO 2：
    # 如果 is_query=True
    # 给文本添加 QUERY_PREFIX
    if is_query:
        text = QUERY_PREFIX + text

    # TODO 3：
    # 调用 model.encode()
    # 设置 normalize_embeddings=True
    vector=model.encode(text,normalize_embeddings=True) #将文本转化为向量

    # TODO 4：
    # 将 NumPy 数组转换为 Python list
    # 提示：.tolist()
    result = vector.tolist()
    return result
