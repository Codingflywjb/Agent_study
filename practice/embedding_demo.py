from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5",
    device="cpu"
)

text = "FastAPI 可以实现大模型流式输出"

vector = model.encode(
    text,
    normalize_embeddings=True
)

print("向量维度：", len(vector))
print("前 5 个数字：", vector[:5])