import uuid
from sentence_transformers import SentenceTransformer
from ingestion.pdf_parser    import parse_pdf
from ingestion.vlm_processor import process_images
from ingestion.chunker       import chunk_pages, add_image_captions
from ingestion.vector_store  import VectorStore
from config import EMBED_MODEL


# Load once at module level — same reason as retriever.py and embedder.py:
# model loading takes ~1 second, so we do it once at startup.
print("Loading embedding model for pipeline...")
_model = SentenceTransformer(EMBED_MODEL)


def run_pipeline(pdf_path: str, skip_vlm: bool = False) -> int:
    """
    End-to-end ingestion: PDF → chunks → embeddings → ChromaDB.

    This is the single function the API (/upload endpoint) will call.
    It wires together every ingestion module in the right order:

        parse_pdf()         — extract text + images from the PDF
        process_images()    — VLM captions for each image (LLaVA via Ollama)
        chunk_pages()       — split long pages into overlapping text chunks
        add_image_captions()— merge caption chunks with text chunks
        embed + store       — embed all chunks and write to ChromaDB

    Args:
        pdf_path : absolute or relative path to the PDF file
        skip_vlm : set True to skip image captioning (useful for text-only PDFs
                   or when Ollama is not running locally)

    Returns:
        total number of chunks stored in ChromaDB
    """
    print(f"\n[Pipeline] Starting ingestion: {pdf_path}")

    # ── Step 1: Parse PDF ────────────────────────────────────────────────────
    # parse_pdf() returns {"pages": [...], "images": [...]}
    print("[Pipeline] Step 1/4 — Parsing PDF...")
    parsed   = parse_pdf(pdf_path)
    pages    = parsed["pages"]
    images   = parsed["images"]
    print(f"           {len(pages)} page(s) with text, {len(images)} image(s) found.")

    # ── Step 2: Caption images with VLM ─────────────────────────────────────
    # LLaVA describes each chart/diagram in plain English so it becomes
    # searchable just like regular text.
    captions = []
    if images and not skip_vlm:
        print(f"[Pipeline] Step 2/4 — Captioning {len(images)} image(s) with LLaVA...")
        captions = process_images(images)
        print(f"           {len(captions)} caption(s) generated.")
    else:
        reason = "skip_vlm=True" if skip_vlm else "no images in document"
        print(f"[Pipeline] Step 2/4 — Skipping VLM ({reason}).")

    # ── Step 3: Chunk ────────────────────────────────────────────────────────
    # Split long pages into overlapping chunks that fit the embedding model's
    # token limit, then append caption chunks (already short — no splitting needed).
    print("[Pipeline] Step 3/4 — Chunking text...")
    chunks = chunk_pages(pages)
    chunks = add_image_captions(chunks, captions)
    print(f"           {len(chunks)} total chunk(s) ready for embedding.")

    if not chunks:
        print("[Pipeline] No chunks produced — nothing to store. Aborting.")
        return 0

    # ── Step 4: Embed and store ──────────────────────────────────────────────
    # Encode all chunks into 384-dimensional vectors, then write to ChromaDB.
    # Using the same model as retriever.py is critical — mismatched models
    # produce vectors in different spaces and make similarity meaningless.
    print("[Pipeline] Step 4/4 — Embedding and storing in ChromaDB...")
    texts     = [c["text"]     for c in chunks]
    metadatas = [
        {
            "source":   c["source"],
            "page_num": c["page_num"],
            "type":     c["type"]
        }
        for c in chunks
    ]
    vectors = _model.encode(texts).tolist()
    ids     = [str(uuid.uuid4()) for _ in chunks]

    vs = VectorStore()
    vs.add(texts=texts, vectors=vectors, metadatas=metadatas, ids=ids)

    total = vs.count()
    print(f"[Pipeline] Done — {len(chunks)} chunk(s) added. "
          f"Collection now has {total} total chunk(s).\n")

    return len(chunks)


# ── Test block ──────────────────────────────────────────────────────────────
# Run with: python -m ingestion.pipeline <path_to_pdf>
# Add --skip-vlm if Ollama is not running.
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.pipeline <path_to_pdf> [--skip-vlm]")
        sys.exit(1)

    path     = sys.argv[1]
    skip_vlm = "--skip-vlm" in sys.argv

    stored = run_pipeline(path, skip_vlm=skip_vlm)
    print(f"Pipeline complete — {stored} chunk(s) stored.")
