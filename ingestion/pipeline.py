from ingestion.pdf_parser    import parse_pdf
from ingestion.vlm_processor import process_images
from ingestion.chunker       import chunk_pages, add_image_captions
from ingestion.embedder      import embed_and_store, get_model


def run_pipeline(pdf_path: str, skip_vlm: bool = False) -> int:
    """
    End-to-end ingestion: PDF → chunks → embeddings → ChromaDB."""
    # This is the single function the API (/upload endpoint) calls.

    # Flow:
    #     parse_pdf()           → extract text pages + raw image bytes
    #     process_images()      → LLaVA captions for each image
    #     chunk_pages()         → split long pages into overlapping chunks
    #     add_image_captions()  → merge caption chunks into text chunks
    #     embed_and_store()     → embed all chunks → write to ChromaDB via VectorStore

    # Args:
    #     pdf_path : path to the PDF file
    #     skip_vlm : True to skip image captioning (no Ollama, or text-only PDF)

    # Returns:
    #     total number of chunks stored
    #
    print(f"\n[Pipeline] Starting ingestion: {pdf_path}")

    # ── Step 1: Parse ────────────────────────────────────────────────────────
    print("[Pipeline] Step 1/4 — Parsing PDF...")
    parsed = parse_pdf(pdf_path)
    pages  = parsed["pages"]
    images = parsed["images"]
    print(f"{len(pages)} page(s) with text, {len(images)} image(s) found.")

    # ── Step 2: Caption images ───────────────────────────────────────────────
    captions = []
    if images and not skip_vlm:
        print(f"[Pipeline] Step 2/4 — Captioning {len(images)} image(s) with LLaVA...")
        captions = process_images(images)
        print(f"{len(captions)} caption(s) generated.")
    else:
        reason = "skip_vlm=True" if skip_vlm else "no images in document"
        print(f"[Pipeline] Step 2/4 — Skipping VLM ({reason}).")

    # ── Step 3: Chunk ────────────────────────────────────────────────────────
    print("[Pipeline] Step 3/4 — Chunking text...")
    chunks = chunk_pages(pages)
    chunks = add_image_captions(chunks, captions)
    print(f"{len(chunks)} total chunk(s) ready for embedding.")

    if not chunks:
        print("[Pipeline] No chunks produced — nothing to store. Aborting.")
        return 0

    # ── Step 4: Embed and store ──────────────────────────────────────────────
    # Delegates entirely to embedder.py — no duplicate model, no direct
    # ChromaDB calls, no uuid logic repeated here.
    print("[Pipeline] Step 4/4 — Embedding and storing in ChromaDB...")
    stored = embed_and_store(chunks)
    print(f"[Pipeline] Done — {stored} chunk(s) stored.\n")
    return stored
