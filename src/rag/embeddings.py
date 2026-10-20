"""Lazy local sentence-transformer embeddings."""
from functools import lru_cache


@lru_cache(maxsize=2)
def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_texts(model, texts):
    model = model if model is not None else load_embedding_model()
    return model.encode(texts, normalize_embeddings=True).tolist()
