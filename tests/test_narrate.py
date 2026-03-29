import pytest
from pathlib import Path
from pipeline.narrate import parse_narration_file, detect_language, get_tts_voice


def test_parse_narration_file(tmp_path):
    narration = tmp_path / "narration.txt"
    narration.write_text("Page one text.\nPage two text.\nPage three text.\n")
    result = parse_narration_file(str(narration))
    assert result == ["Page one text.", "Page two text.", "Page three text."]


def test_parse_narration_file_skips_blank_lines(tmp_path):
    narration = tmp_path / "narration.txt"
    narration.write_text("Page one.\n\nPage two.\n")
    result = parse_narration_file(str(narration))
    assert result == ["Page one.", "Page two."]


def test_detect_language_english():
    lang = detect_language("The rabbit hopped into the garden.")
    assert lang == "en"


def test_get_tts_voice_english():
    voice = get_tts_voice("en")
    assert voice == "nova"


def test_get_tts_voice_fallback():
    voice = get_tts_voice("xx")
    assert voice == "nova"
