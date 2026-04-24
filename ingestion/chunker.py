from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages: list) -> list:
    """
    Takes the 'pages' list from parse_pdf() and splits each page's text
    into smaller, overlapping chunks.
<<<<<<< HEAD
=======

>>>>>>> 4d3477a2ccb7cce11afb29df12df669f7faa5700
    Why chunk at all?
    The embedding model (all-MiniLM-L6-v2) has a ~512 token limit.
    A full PDF page can easily exceed that. Chunking breaks pages into
    pieces the model can embed properly.
<<<<<<< HEAD
=======

>>>>>>> 4d3477a2ccb7cce11afb29df12df669f7faa5700
    Why overlap?
    Without overlap, a sentence sitting at the boundary between two chunks
    would be split in half — losing context. Overlap copies the tail of
    one chunk into the start of the next, so no sentence is orphaned.
<<<<<<< HEAD
    Args:
        pages: list of page dicts from parse_pdf()
               [{"page_num": 1, "text": "...", "source": "file.pdf"}, ...]
=======

    Args:
        pages: list of page dicts from parse_pdf()
               [{"page_num": 1, "text": "...", "source": "file.pdf"}, ...]

>>>>>>> 4d3477a2ccb7cce11afb29df12df669f7faa5700
    Returns:
        list of chunk dicts:
        [{"text": "...", "source": "file.pdf", "page_num": 1, "type": "text"}, ...]
    """
    chunks = []

    for page in pages:
        text     = page["text"]
        source   = page["source"]
        page_num = page["page_num"]

        # Pages shorter than CHUNK_SIZE are stored as a single chunk.
        # No need to split what already fits.
        if len(text) <= CHUNK_SIZE:
            if text.strip():
                chunks.append({
                    "text":     text.strip(),
                    "source":   source,
                    "page_num": page_num,
                    "type":     "text"
                })
            continue

        # Slide a window of CHUNK_SIZE characters across the page text.
        # Each step moves forward by (CHUNK_SIZE - CHUNK_OVERLAP), so the
        # last CHUNK_OVERLAP characters of one chunk become the first
        # CHUNK_OVERLAP characters of the next.
        start = 0
        while start < len(text):
            end        = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text":     chunk_text,
                    "source":   source,
                    "page_num": page_num,
                    "type":     "text"
                })

            if end >= len(text):
                break

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def add_image_captions(chunks: list, image_captions: list) -> list:
    """
    Appends VLM-generated image captions to the chunk list.
<<<<<<< HEAD
=======

>>>>>>> 4d3477a2ccb7cce11afb29df12df669f7faa5700
    Why treat captions separately?
    Captions come from vlm_processor.py as short, pre-formed sentences —
    they don't need splitting. We just tag them as "image_caption" so
    the retriever can label results correctly when showing sources.
<<<<<<< HEAD
=======

>>>>>>> 4d3477a2ccb7cce11afb29df12df669f7faa5700
    Args:
        chunks:         text chunks from chunk_pages()
        image_captions: list of caption dicts from vlm_processor.py
                        [{"page_num": 6, "caption": "...", "source": "file.pdf"}, ...]
<<<<<<< HEAD
=======

>>>>>>> 4d3477a2ccb7cce11afb29df12df669f7faa5700
    Returns:
        combined list — text chunks followed by caption chunks
    """
    for cap in image_captions:
        if cap.get("caption", "").strip():
            chunks.append({
                "text":     cap["caption"].strip(),
                "source":   cap["source"],
                "page_num": cap["page_num"],
                "type":     "image_caption"
            })
    return chunks


# ── Test block ──────────────────────────────────────────────────────────────
# Run with: python -m ingestion.chunker
if __name__ == "__main__":
    # Simulate two pages from parse_pdf()
    fake_pages = [
        {
            "page_num": 1,
            "source":   "sample.pdf",
            "text":     "Short page — fits in one chunk."
        },
        {
            "page_num": 2,
            "source":   "sample.pdf",
            # Long enough to force splitting (>500 chars)
            "text":     (
                "Q3 revenue was $4.2 million, representing 18% year-over-year growth. "
                "The engineering team grew from 45 to 58 people. Net profit margin "
                "improved to 23% in Q3, up from 19% in Q2, due to reduced cloud costs. "
                "Customer churn dropped to 2.1% after the new onboarding program launched "
                "in July. The sales pipeline stands at $12M for Q4, with three enterprise "
                "deals expected to close before December. Marketing spend was reduced by "
                "10% while maintaining lead volume through SEO improvements. "
            ) * 2   # repeat so it's definitely > 500 chars
        }
    ]

    fake_captions = [
        {
            "page_num": 3,
            "source":   "sample.pdf",
            "caption":  "Bar chart showing Q1: $3.1M, Q2: $3.6M, Q3: $4.2M revenue."
        }
    ]

    chunks = chunk_pages(fake_pages)
    chunks = add_image_captions(chunks, fake_captions)

    print(f"Total chunks produced: {len(chunks)}\n")
    for i, c in enumerate(chunks):
        print(f"Chunk {i+1} | page {c['page_num']} | type: {c['type']} | len: {len(c['text'])}")
        print(f"  Preview: {c['text'][:80]}...")
<<<<<<< HEAD
        print()
=======
        print()
>>>>>>> 4d3477a2ccb7cce11afb29df12df669f7faa5700
