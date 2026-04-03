import fitz
from pathlib import Path

def parse_pdf(pdf_path: str) -> dict:
    """ Opens a PDF and extracts two things:
      1. Text from every page  → list of page dicts
      2. Images from every page → list of image dicts

    Why both?
    Text gets embedded directly into ChromaDB.
    Images get sent to LLaVA (VLM) which describes them in words,
    and THOSE descriptions also get embedded into ChromaDB.
    So everything — text and image content — becomes searchable.

    Returns:
        {
          "pages":  [ {page_num, text, source}, ... ],
          "images": [ {page_num, img_bytes, ext, source}, ... ]
        }
        """
    # fitz.open() decodes the PDF binary format into a readable object
    # Think of it like unzipping a zip file
    doc = fitz.open(pdf_path)

    pages  = []   # will hold text from each page
    images = []   # will hold raw image bytes from each page

    for page_num, page in enumerate(doc):

        # --- Extract text ---
        # page.get_text() returns all the text on this page as a string
        # PyMuPDF reads the PDF drawing instructions and converts
        # "draw character A at x=100, y=200" → just the letter "A"
        text = page.get_text()

        # We only store the page if it actually has text
        # Some pages are pure images (like scanned documents)
        # Those will be caught by the image extractor below
        if text.strip():
            pages.append({
                "page_num": page_num + 1,      # humans count from 1, Python from 0
                "text":     text,
                "source":   Path(pdf_path).name  # just the filename, not full path
            })

        # --- Extract images ---
        # page.get_images(full=True) returns a list of every image on this page
        # Each image is identified by an "xref" — a reference number inside the PDF
        for img in page.get_images(full=True):

            xref       = img[0]    # the reference number for this image
            base_image = doc.extract_image(xref)

            # base_image is a dict with:
            #   "image" → raw bytes of the image (like a .jpg or .png file in memory)
            #   "ext"   → the format ("jpeg", "png", etc.)

            images.append({
                "page_num":  page_num + 1,
                "img_bytes": base_image["image"],   # raw bytes, not a file path
                "ext":       base_image["ext"],
                "source":    Path(pdf_path).name
            })

    doc.close()   # always close the file when done — frees memory

    return {
        "pages":  pages,
        "images": images
    }


# ── Test block ─────────────────────────────────────────────────────────────
# This only runs when you do: python ingest/pdf_parser.py
# It does NOT run when other files import parse_pdf
if __name__ == "__main__":
    import sys

    # Check if a PDF path was passed as argument
    # Usage: python ingest/pdf_parser.py sample.pdf
    if len(sys.argv) < 2:
        print("Usage: python ingest/pdf_parser.py <path_to_pdf>")
        print("Example: python ingest/pdf_parser.py sample.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"\n=== Parsing: {pdf_path} ===\n")

    result = parse_pdf(pdf_path)

    # Show what we found
    print(f"✅ Pages with text : {len(result['pages'])}")
    print(f"✅ Images found    : {len(result['images'])}")

    # Preview first page text so you can verify it's correct
    if result["pages"]:
        print(f"\n--- Preview of Page 1 text (first 300 chars) ---")
        print(result["pages"][0]["text"][:300])
        print("...")

    # Show image info
    if result["images"]:
        print(f"\n--- Images found ---")
        for img in result["images"]:
            size_kb = len(img["img_bytes"]) / 1024
            print(f"  Page {img['page_num']} | "
                  f"format: {img['ext']} | "
                  f"size: {size_kb:.1f} KB")
    else:
        print("\n  No images found in this PDF")