import pytest
from pathlib import Path
from PIL import Image, ImageDraw
from pipeline.prepare import normalize_image, prepare_book_images


def make_test_image(path, size=(800, 600), color=(255, 255, 255)):
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 700, 500], fill=(200, 100, 50))
    img.save(path)


def test_normalize_image_produces_1920x1080(tmp_path):
    src = tmp_path / "input.jpg"
    dst = tmp_path / "output.jpg"
    make_test_image(str(src), size=(800, 600))
    normalize_image(str(src), str(dst))
    result = Image.open(str(dst))
    assert result.size == (1920, 1080)


def test_normalize_image_output_is_jpg(tmp_path):
    src = tmp_path / "input.png"
    dst = tmp_path / "output.jpg"
    make_test_image(str(src))
    normalize_image(str(src), str(dst))
    assert dst.exists()


def test_prepare_book_images_creates_normalized_dir(tmp_path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    make_test_image(str(pages_dir / "page01.jpg"))
    make_test_image(str(pages_dir / "page02.jpg"))
    result = prepare_book_images(str(tmp_path))
    normalized_dir = tmp_path / "normalized"
    assert normalized_dir.exists()
    assert len(list(normalized_dir.glob("*.jpg"))) == 2
    assert result == [str(normalized_dir / "page01.jpg"), str(normalized_dir / "page02.jpg")]
