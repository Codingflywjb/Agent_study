import pytest
from practice.rag_similarity import cosine_similarity

def test_cosine_same_direction():
    result = cosine_similarity([3, 4], [6, 8])
    assert result == pytest.approx(1.0)

def test_cosine_orthogonal():
    a=[1,0]
    b=[0,1]
    assert cosine_similarity(a,b) == pytest.approx(0)

def test_cosine_opposite():
    a=[1,0]
    b=[-1,0]
    assert cosine_similarity(a,b) == pytest.approx(-1)


def test_cosine_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity([0,0],[1,2])

def test_cosine_invalid_dimensions():
    with pytest.raises(ValueError):
        cosine_similarity([1,2],[1])

def test_cosine_empty_vector():
    with pytest.raises(ValueError):
        cosine_similarity([],[1,2])

