import chromadb
from config import CHROMA_PATH, COLLECTION_NAME


class VectorStore:
    """
    A thin wrapper around ChromaDB that centralises all database access
    for the ingestion pipeline.
    Why a class instead of bare chromadb calls scattered around?
    If you later swap ChromaDB for Pinecone or Milvus (the original plan),
    you only change this one file — embedder.py and pipeline.py stay the same.
    Usage:
        vs = VectorStore()
        vs.add(texts, vectors, metadatas, ids)
        print(vs.count())
    """

    def __init__(self):
        # PersistentClient saves data to disk at CHROMA_PATH.
        # cosine space is set at collection creation and must match
        # how retriever.py interprets distances (1 - distance = similarity).
        self._db  = chromadb.PersistentClient(path=CHROMA_PATH)
        self._col = self._db.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, texts: list, vectors: list, metadatas: list, ids: list) -> None:
        """
        Store a batch of chunks in ChromaDB.
        Args:
            texts     : raw text strings (what the retriever returns to the agent)
            vectors   : pre-computed embeddings — list of 384-float lists
            metadatas : list of dicts with keys: source, page_num, type
            ids       : unique string ID per chunk (use uuid4)
        """
        self._col.add(
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids
        )

    def count(self) -> int:
        """Return the total number of chunks currently stored."""
        return self._col.count()

    def clear(self) -> None:
        """
        Delete and recreate the collection (wipes all stored chunks).
        Use this before loading real documents to remove dummy/test data.
        Warning: irreversible — all vectors are gone after this call.
        """
        try:
            self._db.delete_collection(COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' cleared.")
        except Exception:
            print(f"Collection '{COLLECTION_NAME}' did not exist — nothing to clear.")

        self._col = self._db.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Fresh collection '{COLLECTION_NAME}' ready.")


# ── Test block ──────────────────────────────────────────────────────────────
# Run with: python -m ingestion.vector_store
if __name__ == "__main__":
    import uuid

    vs = VectorStore()
    print(f"Collection has {vs.count()} chunks before test.\n")

    # Insert one dummy chunk
    vs.add(
        texts=["VectorStore test chunk — Q3 revenue was $4.2M."],
        vectors=[[0.1] * 384],   # fake vector, not a real embedding
        metadatas=[{"source": "test.pdf", "page_num": 1, "type": "text"}],
        ids=[str(uuid.uuid4())]
    )
    print(f"After insert: {vs.count()} chunks in collection.")
    print("VectorStore test passed.")