import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL, TOP_K


# Why load the model at module level (outside any function)?
# Because loading a model takes ~1 second. If we loaded it
# inside retrieve() every call would be 1 second slower.
# Loading once at startup = fast for every subsequent call.
print("Loading embedding model for retriever...")
_embedder = SentenceTransformer(EMBED_MODEL)


def embed_query(text: str) -> list:
    """
    Convert a text query into a vector (list of 384 numbers).

    Why do we embed the query with the SAME model used during ingestion?
    Because embedding models create a shared vector space.
    "Q3 revenue" and "how much did the company earn" both point
    to the same region of that space — but ONLY if embedded
    with the same model. Mixing models = meaningless comparisons.
    """
    vector = _embedder.encode(text)
    return vector.tolist()   # ChromaDB needs a plain list, not numpy array


def retrieve(query: str, top_k: int = TOP_K) -> list:
    """
    The core function. Takes a plain English question,
    returns the top_k most semantically similar chunks.

    How it works step by step:
    1. Embed the query → 384-number vector
    2. ChromaDB compares that vector against ALL stored vectors
       using cosine similarity (measures angle between vectors)
    3. Returns the closest matches — chunks about similar topics

    Args:
        query  : the user's question in plain English
        top_k  : how many chunks to return (default 5 from config)

    Returns:
        list of dicts, each containing:
        - text       : the actual chunk content
        - source     : which file it came from
        - page       : which page number
        - type       : "text" or "image_caption"
        - similarity : 0.0 to 1.0 (1.0 = identical, 0.0 = unrelated)
    """

    # Connect to ChromaDB — same path as where dummy data was stored
    db  = chromadb.PersistentClient(path=CHROMA_PATH)
    col = db.get_collection(COLLECTION_NAME)

    # Embed the query into a vector
    query_vector = embed_query(query)

    # Search ChromaDB
    # include= tells ChromaDB what to return alongside the results
    # distances = how far each result is from the query vector
    results = col.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # ChromaDB returns nested lists because it supports batch queries.
    # We only sent 1 query, so we take index [0] to unwrap the batch.
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Why 1 - distance?
    # ChromaDB returns DISTANCE (lower = more similar).
    # We convert to SIMILARITY (higher = more similar) because
    # 0.95 similarity is more intuitive than 0.05 distance.
    chunks = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append({
            "text":       doc,
            "source":     meta.get("source",   "unknown"),
            "page":       meta.get("page_num", "?"),
            "type":       meta.get("type",     "text"),
            "similarity": round(1 - dist, 3)
        })

    return chunks


# ── Test block ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n=== Retriever Test ===\n")

    # Test queries — these should find relevant chunks
    # even though the wording is different from what's stored
    test_queries = [
        "how much money did the company make?",
        "how many people work here?",
        "what does the revenue chart show?",    # should find image_caption chunk
        "what is the customer churn rate?",
    ]

    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 50)

        results = retrieve(query, top_k=2)

        for i, chunk in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"    Similarity : {chunk['similarity']}")
            print(f"    Type       : {chunk['type']}")
            print(f"    Source     : {chunk['source']} p.{chunk['page']}")
            print(f"    Text       : {chunk['text'][:100]}...")
        print()

    print("=== Done ===")

    #To run this, type in terminal:
    # python -m retrieval.retriever