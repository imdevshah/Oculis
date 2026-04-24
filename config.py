import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_PATH  = "./chroma_db"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 10
SELFCHECK_SAMPLES = 2
CONFIDENCE_THRESHOLD = 0.7

# All free models
EMBED_MODEL          = "BAAI/bge-m3"   # runs locally, no API
ANSWER_MODEL         = "meta-llama/llama-4-scout-17b-16e-instruct"  # Groq, free
CAPTION_MODEL        = "minicpm-v"               # Ollama, local, free
CHECK_MODEL          = "llama-3.3-70b-versatile"     # Groq, free + fast

# Token budget constants — prevents 413 errors across all endpoints
MAX_CONTEXT_CHARS = 5000   # ~300 tokens — enough to verify answers
MAX_CHUNK_CHARS   = 600    # per chunk returned by rag_search
# Test block:
if __name__ == "__main__":
    print("=== Config Test ===")

    # Test 1: .env loaded correctly
    if GROQ_API_KEY:
        print(f" GROQ_API_KEY loaded  (starts with: {GROQ_API_KEY[:8]}...)")
    else:
        print(" GROQ_API_KEY missing — check your .env file")

    # Test 2: Groq API actually works
    print("\nTesting Groq connection...")
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        r = client.chat.completions.create(
            model=CHECK_MODEL,
            messages=[{"role": "user", "content": "say hi in one word"}],
            max_tokens=5
        )
        print(f" Groq works — model said: {r.choices[0].message.content}")
    except Exception as e:
        print(f" Groq failed: {e}")

    # Test 3: Sentence transformers work
    print("\nTesting local embeddings...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        vec = model.encode("hello world")
        print(f" Embeddings work — vector size: {len(vec)}")
    except Exception as e:
        print(f" Embeddings failed: {e}")

    # Test 4: ChromaDB works
    print("\nTesting ChromaDB...")
    try:
        import chromadb
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        col = db.get_or_create_collection("test")
        col.add(documents=["test doc"], ids=["test-1"])
        print(f" ChromaDB works — items in collection: {col.count()}")
    except Exception as e:
        print(f" ChromaDB failed: {e}")

    # Test 5: Ollama works
    print("\nTesting Ollama...")
    try:
        import ollama
        r = ollama.chat(
            model=CAPTION_MODEL,
            messages=[{"role": "user", "content": "say hi in one word"}]
        )
        print(f" Ollama works — model said: {r['message']['content']}")
    except Exception as e:
        print(f" Ollama failed: {e}")

    print("\n=== Config Test Complete ===")
