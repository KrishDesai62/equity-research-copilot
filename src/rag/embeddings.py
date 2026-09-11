"""
Embed chunks for storage in the vector store. Local sentence-transformers
model by default (free) — see config/settings.yaml `rag.embedding_model`.
"""
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

def load_embedding_model(model_name: str):
    """
    TODO: from sentence_transformers import SentenceTransformer;
    return SentenceTransformer(model_name)
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
    )


def embed_texts(model, texts: List[str]) -> List[List[float]]:
    return model.embed_documents(texts)
    


##CHANGE INTO VOYAGE AI API ONCE I AM GOOD WITH TESTING