import uuid
from sentence_transformers import SentenceTransformer
from ingestion.vector_store import VectorStore
from config import EMBED_MODEL

# Load once at module level — pipeline.py imports this and reuses it.
# Only ONE copy of the model lives in memory across the whole process.
print("Loading embedding model for embedder...")
_model = SentenceTransformer(EMBED_MODEL)


def get_model():
    """
    Returns the shared model instance.
    pipeline.py calls this instead of loading its own copy.
    """
    return _model


def embed_and_store(chunks: list) -> int:
    """
    Embeds a list of chunk dicts and stores them via VectorStore.

    Why delegate to VectorStore instead of calling ChromaDB directly?
    VectorStore is the single place that knows about the database.
    If we swap ChromaDB for Pinecone tomorrow, only vector_store.py
    changes — this function stays identical.

    Args:
        chunks: list of chunk dicts from chunker.py
                [{"text": "...", "source": "...", "page_num": 1, "type": "text"}, ...]
    Returns:
        number of chunks stored
    """
    if not chunks:
        print("No chunks to embed — skipping.")
        return 0

    texts     = [c["text"]     for c in chunks]
    metadatas = [
        {
            "source":   c["source"],
            "page_num": c["page_num"],
            "type":     c["type"]
        }
        for c in chunks
    ]

    # encode() returns numpy array shape (n_chunks, 384).
    # .tolist() converts to plain Python list — ChromaDB requires that.
    vectors = _model.encode(texts).tolist()

    # uuid4() generates a unique random ID per chunk.
    # Collision probability is negligible (2^122 possible values).
    ids = [str(uuid.uuid4()) for _ in chunks]

    # Route through VectorStore — NOT direct chromadb calls.
    # This is the key fix: one code path to the database, not two.
    vs = VectorStore()
    vs.add(texts=texts, vectors=vectors, metadatas=metadatas, ids=ids)

    total = vs.count()
    print(f"Stored {len(chunks)} chunks. Collection now has {total} total.")
    return len(chunks)


def clear_collection() -> None:
    """
    Wipes and recreates the ChromaDB collection via VectorStore.
    Use this before loading real documents to remove dummy/test data.
    """
    vs = VectorStore()
    vs.clear()
