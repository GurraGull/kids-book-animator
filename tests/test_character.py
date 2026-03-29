import pytest
from pathlib import Path
from PIL import Image
from pipeline.character import remove_background, ACTION_TO_MOTION


def test_remove_background_returns_png(tmp_path):
    src = tmp_path / "character.jpg"
    dst = tmp_path / "character_nobg.png"
    img = Image.new("RGB", (100, 100), (200, 100, 50))
    img.save(str(src))
    result = remove_background(str(src), str(dst))
    assert Path(result).suffix == ".png"
    assert Path(result).exists()


def test_action_to_motion_walk():
    assert ACTION_TO_MOTION["walk"] == "walks_in_place"


def test_action_to_motion_wave():
    assert ACTION_TO_MOTION["wave"] == "wave_hello"


def test_action_to_motion_jump():
    assert ACTION_TO_MOTION["jump"] == "jumping"
