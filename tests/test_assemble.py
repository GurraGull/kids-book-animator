import pytest
import subprocess
from pipeline.assemble import get_audio_duration, build_subtitle_filter


def test_get_audio_duration_returns_float(tmp_path):
    mp3_path = str(tmp_path / "silent.mp3")
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
        "-t", "2.5", "-q:a", "9", "-acodec", "libmp3lame", mp3_path, "-y"
    ], check=True, capture_output=True)
    duration = get_audio_duration(mp3_path)
    assert 2.0 < duration < 3.0


def test_build_subtitle_filter_contains_text():
    filt = build_subtitle_filter("Hello world", 1920, 1080)
    assert "Hello world" in filt
    assert "fontsize" in filt
    assert "shadowx" in filt


def test_build_subtitle_filter_escapes_apostrophe():
    filt = build_subtitle_filter("it's a book", 1920, 1080)
    assert "it" in filt
