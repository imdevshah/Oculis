from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages: list) -> list:
    """
    Takes the 'pages' list from parse_pdf() and splits each page's text
    into smaller, overlapping chunks.


    Why chunk at all?
    The embedding model (all-MiniLM-L6-v2) has a ~512 token limit.
    A full PDF page can easily exceed that. Chunking breaks pages into
    pieces the model can embed properly.


    Why overlap?
    Without overlap, a sentence sitting at the boundary between two chunks
    would be split in half — losing context. Overlap copies the tail of
    one chunk into the start of the next, so no sentence is orphaned.
    Args:
        pages: list of page dicts from parse_pdf()
               [{"page_num": 1, "text": "...", "source": "file.pdf"}, ...]

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
    Why treat captions separately?
    Captions come from vlm_processor.py as short, pre-formed sentences —
    they don't need splitting. We just tag them as "image_caption" so
    the retriever can label results correctly when showing sources.

    Args:
        chunks:         text chunks from chunk_pages()
        image_captions: list of caption dicts from vlm_processor.py
                        [{"page_num": 6, "caption": "...", "source": "file.pdf"}, ...]



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
