import re
import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL, TOP_K

print("Loading embedding model for retriever...")
_embedder = SentenceTransformer(EMBED_MODEL)


def embed_query(text: str) -> list:
    return _embedder.encode(text).tolist()


def retrieve(query: str, top_k: int = TOP_K) -> list:
    db  = chromadb.PersistentClient(path=CHROMA_PATH)
    col = db.get_collection(COLLECTION_NAME)

    query_vector = embed_query(query)
    query_lower  = query.lower()

    is_definition_query = any(p in query_lower for p in ["what is", "what are", "define","explain", "overview", "introduction"])


    query_terms = [w for w in re.findall(r'\w+', query_lower)if len(w) > 3]


    precise_refs = []
    for m in re.finditer(
        r'\b(table|figure|fig|section|appendix)\s*\.?\s*(\d+)\b',
        query_lower
    ):
        precise_refs.append({m.group(1), m.group(2)})

    # Fetch large candidate pool — tables of numbers have low base similarity
    # We need page 16 to even appear before we can boost it
    total_chunks = col.count()
    fetch_k      = min(total_chunks, max(top_k * 6, 30))

    results = col.query(
        query_embeddings=[query_vector],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # # Named proper nouns — skip common words
    # stopwords = {
    #     "what", "which", "where", "when", "who", "how", "why",
    #     "according", "shown", "many", "does", "about", "from",
    #     "each", "give", "tell", "list", "describe", "explain",
    #     "the", "and", "for", "with", "that", "this", "are",
    #     "were", "was", "have", "has", "been", "into", "page",
    #     "information", "samples", "generated", "pattern", "patterns",
    #     "number", "numbers", "per", "total", "show", "shows",
    #     "how", "many", "were", "each", "hallucination", "table",
    #     "figure", "section", "appendix"
    # }
    # named_terms = []
    # for i, word in enumerate(query.split()):
    #     clean = re.sub(r'[^a-zA-Z]', '', word).lower()
    #     if (clean and len(clean) > 2 and i > 0
    #             and word[0].isupper()
    #             and clean not in stopwords):
    #         named_terms.append(clean)

    chunks = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        doc_lower = doc.lower()
        base_sim  = max(0.0, 1.0 - (dist / 2.0))
        boost     = 0.0

        page_num = meta.get("page_num", 999)
        try:
            page_num = int(page_num)
        except:
            page_num = 999

        if is_definition_query and page_num <= 3:
            boost += 0.6

        term_matches = sum(1 for t in query_terms if t in doc_lower)
        boost += term_matches* 0.05

            
        # for ref in precise_refs:
        #     pattern = (rf'\b{re.escape(ref["type"])}'
        #                rf'\s+{re.escape(ref["number"])}\b')
        #     if re.search(pattern, doc_lower, re.IGNORECASE | re.DOTALL):
        #         boost += 1.0
        #         break
        for r_type, r_num in precise_refs:
            pattern = rf'\b{r_type}\s+{r_num}\b'
            if re.search(pattern, doc_lower):
                boost += 1.0
                break

        #  Named term boost
        # for term in named_terms:
        #     if term in doc_lower:
        #         boost += 0.08
        #         break

        #  Page number boost
        page_match = re.search(r'\bpage\s*(\d+)\b', query_lower)
        if page_match:
            if str(meta.get("page_num")) == page_match.group(1):
                boost += 0.5  

        #  Visual content boost
        visual_words = {"chart", "diagram", "figure", "image",
                        "plot", "graph", "visual", "picture"}
        if any(w in query_lower for w in visual_words):
            if meta.get("type") == "image_caption":
                boost += 2.0
        
        final_score = base_sim +  boost

        chunks.append({
            "text":       doc,
            "source":     meta.get("source","unknown"),
            "page":       page_num,
            "type":       meta.get("type","text"),
            "similarity": round(final_score, 3)
        })

    chunks.sort(key=lambda x: x["similarity"], reverse=True)
    return chunks[:top_k]
