from math import sqrt

def cosine_similarity(a:list[float], b:list[float]) -> float:
    if not a or not b:
        raise ValueError("a and b must be non-empty")
    if len(a) != len(b):
        raise ValueError("a and b must have same length")
    dot_product = sum([a[i] * b[i] for i in range(len(a))])
    len_a = sqrt(sum(a[i]*a[i] for i in range(len(a))))
    len_b = sqrt(sum(b[i]*b[i] for i in range(len(b))))
    if len_a == 0 or len_b == 0:
        raise ValueError("向量长度不为0")
    return dot_product / (len_a * len_b)
