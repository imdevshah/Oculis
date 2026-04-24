import fitz
from pathlib import Path

def parse_pdf(pdf_path: str) -> dict:
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

    #     # --- Extract images ---
        has_embedded_images = len(page.get_images(full=True)) > 0
        is_short_text_page  = len(text.strip()) < 200  # likely a figure page

        if has_embedded_images or is_short_text_page:
            mat       = fitz.Matrix(1.5, 1.5)
            pix       = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            images.append({
                "page_num":  page_num + 1,
                "img_bytes": img_bytes,
                "ext":       "png",
                "source":    Path(pdf_path).name
            })

    doc.close()
    return {"pages": pages, "images": images}