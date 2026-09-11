import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb

# Configure production logging
logger = logging.getLogger("VectorStore")

class VectorStore:
    """
    Production-grade thin wrapper around ChromaDB providing persistent storage, 
    deterministic ID generation, and metadata-filtered similarity queries.
    """
    def __init__(self, persist_directory: str, collection_name: str = "filings"):
        self.persist_path = Path(persist_directory)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_path))
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Initialized ChromaDB collection '{collection_name}' at path: {self.persist_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise RuntimeError(f"Vector store initialization failed: {e}")

    def add_chunks(self, chunks: List[Any], embeddings: List[List[float]]) -> None:
        """
        Adds text chunks and their corresponding embedding vectors into the vector store 
        with deterministic MD5 hashed IDs to prevent duplicate entries.
        """
        if not chunks or not embeddings:
            logger.warning("Empty chunks or embeddings list provided to add_chunks.")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Length mismatch: {len(chunks)} chunks provided for {len(embeddings)} embeddings.")

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            # Support both dataclass objects and dictionary structures gracefully
            if hasattr(chunk, "ticker"):
                ticker = getattr(chunk, "ticker", "UNKNOWN")
                filing_date = getattr(chunk, "filing_date", "1970-01-01")
                section = getattr(chunk, "section", "UNKNOWN")
                text = getattr(chunk, "text", "")
                filing_type = getattr(chunk, "filing_type", "UNKNOWN")
                fiscal_period = getattr(chunk, "fiscal_period", "UNKNOWN")
            else:
                ticker = chunk.get("ticker", "UNKNOWN")
                filing_date = chunk.get("filing_date", "1970-01-01")
                section = chunk.get("section", "UNKNOWN")
                text = chunk.get("text", "")
                filing_type = chunk.get("filing_type", "UNKNOWN")
                fiscal_period = chunk.get("fiscal_period", "UNKNOWN")

            # Deterministic unique hash string based on immutable business identifiers and index
            unique_string = f"{ticker}_{filing_date}_{section}_{idx}_{hash(text[:50])}"
            chunk_id = hashlib.md5(unique_string.encode("utf-8")).hexdigest()

            ids.append(chunk_id)
            documents.append(text)
            
            # Ensure metadata values comply with ChromaDB schema requirements (str, int, float, bool)
            metadatas.append({
                "ticker": str(ticker),
                "filing_type": str(filing_type),
                "fiscal_period": str(fiscal_period),
                "filing_date": str(filing_date),
                "section_title": str(section)  # Aligns with retrieval.py mapping expectation
            })

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Successfully inserted {len(ids)} chunks into vector store collection.")
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB collection: {e}")
            raise RuntimeError(f"Vector store batch insertion failed: {e}")

    def query(self, query_embedding: List[float], k: int = 5, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes a vector similarity query with optional metadata filtering.
        """
        if not query_embedding:
            raise ValueError("Query embedding vector cannot be empty.")

        try:
            # Chroma expects query_embeddings as a list of lists
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where if where else None
            )
            return results
        except Exception as e:
            logger.error(f"Error executing vector store query with filter {where}: {e}")
            raise RuntimeError(f"Vector store query failed: {e}")