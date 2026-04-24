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
                "ACT AS A DATA EXTRACTOR. Look at this image from page " + str(page_num) + ". "
                "IF THERE IS A TABLE: Write out every number and header. "
                "IF THERE IS A CHART: Describe the X-axis, Y-axis, and the data points. "
                "IF THERE IS A DIAGRAM: Describe the labels and how they connect. "
                "Output your description starting with: 'This visual on page " + str(page_num) + " shows...'"
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
            print(f"Done: {cap['caption'][:80]}...")
        except Exception as e:
            # Log the failure but keep going — partial results are better than none.
            print(f"Warning: could not caption image on page {img['page_num']}: {e}")

    return captions
