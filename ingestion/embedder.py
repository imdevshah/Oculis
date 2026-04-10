import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL


# Load the model once at module level, not inside the function.
# Reason: loading a SentenceTransformer takes ~1 second.
# Doing it once at startup means every embed_and_store() call is fast.
print("Loading embedding model for embedder...")
_model = SentenceTransformer(EMBED_MODEL)


def embed_and_store(chunks: list) -> int:
    """
    Embeds a list of chunk dicts and stores them in ChromaDB.

    Why pre-embed at ingestion time?
    At query time you want fast answers. If we embedded on the fly,
    every search would call the model — slow. Pre-embedding means
    ChromaDB does instant vector comparisons at query time.

    This function uses the SAME model as retriever.py (all-MiniLM-L6-v2).
    That's critical: query vectors and document vectors must come from
    the same model, or cosine similarity comparisons are meaningless.

    Args:
        chunks: list of chunk dicts from chunker.py
                [{"text": "...", "source": "...", "page_num": 1, "type": "text"}, ...]

    Returns:
        number of chunks stored
    """
    if not chunks:
        print("No chunks to embed — skipping.")
        return 0

    # Connect to ChromaDB on disk (creates the folder if it doesn't exist).
    # cosine space means similarity = 1 - cosine_distance,
    # which is what retriever.py assumes when it does `1 - dist`.
    db  = chromadb.PersistentClient(path=CHROMA_PATH)
    col = db.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    texts     = [c["text"]     for c in chunks]
    metadatas = [
        {
            "source":   c["source"],
            "page_num": c["page_num"],
            "type":     c["type"]
        }
        for c in chunks
    ]

    # encode() returns a numpy array of shape (n_chunks, 384).
    # .tolist() converts it to a plain Python list — ChromaDB requires that.
    vectors = _model.encode(texts).tolist()

    # Each chunk needs a unique ID so ChromaDB can store and look it up.
    # uuid4() generates a random ID — collision probability is negligible.
    ids = [str(uuid.uuid4()) for _ in chunks]

    col.add(
        documents=texts,
        embeddings=vectors,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Stored {len(chunks)} chunks. Collection now has {col.count()} total.")
    return len(chunks)


def clear_collection() -> None:
    """
    Deletes and recreates the ChromaDB collection.

    Use this when your teammate's real ingestion pipeline is ready
    and you want to wipe the dummy data before loading real documents.

    Warning: this permanently deletes all stored chunks.
    """
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        db.delete_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' deleted.")
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' did not exist — nothing to delete.")

    db.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Fresh empty collection '{COLLECTION_NAME}' created.")


# ── Test block ──────────────────────────────────────────────────────────────
# Run with: python -m ingestion.embedder
if __name__ == "__main__":
    from ingestion.chunker import chunk_pages, add_image_captions

    fake_pages = [
        {
            "page_num": 1,
            "source":   "embedder_test.pdf",
            "text":     "Embedder test chunk one. Revenue was $4.2M in Q3."
        },
        {
            "page_num": 2,
            "source":   "embedder_test.pdf",
            "text":     "Embedder test chunk two. Headcount grew to 142 employees."
        }
    ]

    fake_captions = [
        {
            "page_num": 3,
            "source":   "embedder_test.pdf",
            "caption":  "Bar chart: Q1 $3.1M, Q2 $3.6M, Q3 $4.2M."
        }
    ]

    chunks = chunk_pages(fake_pages)
    chunks = add_image_captions(chunks, fake_captions)

    print(f"\nEmbedding {len(chunks)} chunks...\n")
    stored = embed_and_store(chunks)
    print(f"\nDone — {stored} chunks embedded and stored in ChromaDB.")
