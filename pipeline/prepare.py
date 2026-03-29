from pathlib import Path
from PIL import Image, ImageOps

TARGET_SIZE = (1920, 1080)


def normalize_image(src_path: str, dst_path: str) -> None:
    """Resize and letterbox an image to 1920x1080, save as JPEG."""
    img = Image.open(src_path).convert("RGB")
    img = ImageOps.exif_transpose(img)  # fix rotation from phone cameras
    img.thumbnail(TARGET_SIZE, Image.LANCZOS)
    # Letterbox: paste onto black 1920x1080 canvas
    canvas = Image.new("RGB", TARGET_SIZE, (0, 0, 0))
    x = (TARGET_SIZE[0] - img.width) // 2
    y = (TARGET_SIZE[1] - img.height) // 2
    canvas.paste(img, (x, y))
    canvas.save(dst_path, "JPEG", quality=95)


def extract_pdf_pages(pdf_path: str, output_dir: str) -> list:
    """
    Extract each page of a PDF as a JPEG image.
    Returns sorted list of output file paths.
    Requires poppler (brew install poppler on macOS).
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image is required for PDF support. Install with:\n"
            "  pip3 install pdf2image\n"
            "  brew install poppler"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=150)
    paths = []
    for i, page in enumerate(pages):
        dst = out_dir / f"page{i+1:02d}.jpg"
        page.save(str(dst), "JPEG", quality=92)
        paths.append(str(dst))
        print(f"  Extracted PDF page {i+1}/{len(pages)} → {dst.name}")
    return paths


def prepare_book_images(book_dir: str) -> list:
    """
    Normalize all images in <book_dir>/pages/ to <book_dir>/normalized/.
    Also handles a single book.pdf in the book folder — extracts pages first.
    Returns sorted list of output file paths.
    """
    book_path = Path(book_dir)
    pages_dir = book_path / "pages"
    normalized_dir = book_path / "normalized"
    normalized_dir.mkdir(exist_ok=True)

    # If no pages/ dir but a PDF exists, extract it first
    pdf_files = list(book_path.glob("*.pdf"))
    if not pages_dir.exists() and pdf_files:
        print(f"  Found PDF: {pdf_files[0].name} — extracting pages...")
        pages_dir.mkdir(exist_ok=True)
        extract_pdf_pages(str(pdf_files[0]), str(pages_dir))

    if not pages_dir.exists():
        raise FileNotFoundError(
            f"No pages/ directory found in {book_dir}. "
            "Create a pages/ folder with your book images, or place a .pdf file in the book folder."
        )

    image_files = sorted(
        [f for f in pages_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    if not image_files:
        raise FileNotFoundError(f"No image files found in {pages_dir}")

    output_paths = []
    for img_file in image_files:
        dst = normalized_dir / (img_file.stem + ".jpg")
        normalize_image(str(img_file), str(dst))
        output_paths.append(str(dst))
        print(f"  Prepared: {img_file.name} → {dst.name}")
    return output_paths
