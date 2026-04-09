# create_dummy_data.py
# Run this ONCE to fill ChromaDB with fake chunks so I can
# build and test your agent WITHOUT waiting for my teammate.
# When your friend's real ingestion is ready, just delete
# the chroma_db/ folder and use theirs instead.

import chromadb
import uuid
from sentence_transformers import SentenceTransformer
from config import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL

print("Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)

db  = chromadb.PersistentClient(path=CHROMA_PATH)  #PersistentClient: This tells ChromaDB to save the data to your hard drive at the CHROMA_PATH (e.g., ./chroma_db). If you used an "Ephemeral" client, the data would vanish when the script stops.

col = db.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}) #This tells the database to use "Cosine Similarity" to find matches.

# Fake document chunks — pretend these came from a real PDF
fake_chunks = [
    {
        "text": "Q3 revenue was $4.2 million, representing 18% "
                "year-over-year growth driven by strong SaaS subscriptions.",
        "source": "q3_report.pdf", "page_num": 3, "type": "text"
    },
    {
        "text": "The company employs 142 people across 3 offices. "
                "Engineering is the largest department with 58 employees.",
        "source": "q3_report.pdf", "page_num": 5, "type": "text"
    },
    {
        "text": "Net profit margin improved to 23% in Q3, up from "
                "19% in Q2, due to reduced cloud infrastructure costs.",
        "source": "q3_report.pdf", "page_num": 4, "type": "text"
    },
    {
        "text": "[Chart on page 6 of q3_report.pdf]: Bar chart showing "
                "quarterly revenue. Q1: $3.1M, Q2: $3.6M, Q3: $4.2M. "
                "Clear upward trend across all three quarters.",
        "source": "q3_report.pdf", "page_num": 6, "type": "image_caption"
    },
    {
        "text": "Customer churn rate dropped to 2.1% this quarter. "
                "Key retention driver was the new onboarding program "
                "launched in July.",
        "source": "q3_report.pdf", "page_num": 7, "type": "text"
    },
]

print(f"Embedding and storing {len(fake_chunks)} dummy chunks...")

texts     = [c["text"] for c in fake_chunks] #A "list comprehension" that extracts just the sentences from your fake data.
vectors   = model.encode(texts).tolist() #The model turns your sentences into long lists of numbers. We convert them to a list() so ChromaDB can store them. Each list of numbers is called an "embedding" and captures the meaning of the sentence in a way that the database can understand and compare.
metadatas = [{"source": c["source"],
              "page_num": c["page_num"],
              "type": c["type"]} for c in fake_chunks] #This stores the "extra" info (Source file, Page number). When your AI finds an answer, it uses this to say: "I found this on Page 3 of q3_report.pdf."
ids       = [str(uuid.uuid4()) for _ in fake_chunks] #This creates a unique ID for each chunk. It's like giving each piece of information its own name tag, so we can keep track of it in the database.

col.add(documents=texts, embeddings=vectors,
        metadatas=metadatas, ids=ids) #This line actually puts everything into the database. It says: "Hey ChromaDB, here are the sentences (documents), their meanings (embeddings), and the extra info (metadatas). Please store them under these unique IDs."

print(f" Done — {col.count()} chunks now in ChromaDB")
print("You can now build and test your agent against this data.")
print("When your friend's real data is ready, delete chroma_db/ and use theirs.")