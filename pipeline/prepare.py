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


def prepare_book_images(book_dir: str) -> list:
    """
    Normalize all images in <book_dir>/pages/ to <book_dir>/normalized/.
    Returns sorted list of output file paths.
    """
    book_path = Path(book_dir)
    pages_dir = book_path / "pages"
    normalized_dir = book_path / "normalized"
    normalized_dir.mkdir(exist_ok=True)

    image_files = sorted(
        [f for f in pages_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    output_paths = []
    for img_file in image_files:
        dst = normalized_dir / (img_file.stem + ".jpg")
        normalize_image(str(img_file), str(dst))
        output_paths.append(str(dst))
        print(f"  Prepared: {img_file.name} → {dst.name}")
    return output_paths
