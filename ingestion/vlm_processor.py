import base64
import ollama
from config import CAPTION_MODEL


def caption_image(img_bytes: bytes, page_num: int, source: str) -> dict:
    """
    Sends a single image to LLaVA (via Ollama) and returns its caption.

    Why LLaVA?
    LLaVA is a Vision-Language Model — it understands both images and text.
    When given a chart or table from a PDF, it can describe the data in
    plain English, which we then embed into ChromaDB just like any text chunk.
    This makes visual content searchable by the agent.

    Why Ollama?
    Ollama runs LLaVA locally for free — no API key or cost per image.

    Args:
        img_bytes : raw image bytes from parse_pdf() (img["img_bytes"])
        page_num  : page the image came from (preserved in metadata)
        source    : PDF filename (preserved in metadata)

    Returns:
        dict with keys: page_num, source, caption
    """
    # Ollama's API expects the image as a base64-encoded string.
    # base64 converts binary bytes → ASCII text that can be sent in JSON.
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    response = ollama.chat(
        model=CAPTION_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "You are analyzing a page extracted from a business PDF document. "
                "Describe this image in detail. "
                "If it contains a chart, table, or graph: extract all labels, axis titles, "
                "and numerical values you can see. "
                "If it is a diagram or photo: describe what it shows. "
                "Be precise and concise."
            ),
            "images": [img_b64]
        }]
    )

    caption = response["message"]["content"].strip()

    return {
        "page_num": page_num,
        "source":   source,
        "caption":  caption
    }


def process_images(images: list) -> list:
    """
    Runs caption_image() over every image extracted by parse_pdf().

    Failures are caught per-image so one bad image doesn't abort
    the entire ingestion of a document.

    Args:
        images: the "images" list from parse_pdf()
                [{"page_num": 1, "img_bytes": b"...", "ext": "jpeg", "source": "file.pdf"}, ...]

    Returns:
        list of caption dicts:
        [{"page_num": 1, "source": "file.pdf", "caption": "..."}, ...]
    """
    captions = []

    for i, img in enumerate(images):
        print(f"  Captioning image {i + 1}/{len(images)} (page {img['page_num']})...")
        try:
            cap = caption_image(
                img_bytes=img["img_bytes"],
                page_num=img["page_num"],
                source=img["source"]
            )
            captions.append(cap)
            print(f"    Done: {cap['caption'][:80]}...")
        except Exception as e:
            # Log the failure but keep going — partial results are better than none.
            print(f"    Warning: could not caption image on page {img['page_num']}: {e}")

    return captions


# ── Test block ──────────────────────────────────────────────────────────────
# Requires Ollama running locally with LLaVA pulled:
#   ollama pull llava
#   ollama serve
# Run with: python -m ingestion.vlm_processor <path_to_pdf>
if __name__ == "__main__":
    import sys
    from ingestion.pdf_parser import parse_pdf

    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.vlm_processor <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"\nParsing: {pdf_path}")
    result = parse_pdf(pdf_path)

    images = result["images"]
    print(f"Found {len(images)} image(s). Sending to LLaVA...\n")

    captions = process_images(images)

    print(f"\n=== {len(captions)} caption(s) generated ===")
    for cap in captions:
        print(f"\nPage {cap['page_num']}:")
        print(f"  {cap['caption']}")
