import os
from pathlib import Path
from langdetect import detect as langdetect_detect
from openai import OpenAI

# Voice selection by language — OpenAI TTS voices
VOICE_MAP = {
    "en": "nova",    # warm, child-friendly English
    "sv": "nova",    # Swedish — nova works well
    "default": "nova",
}


def parse_narration_file(narration_path: str) -> list:
    """Read narration.txt and return list of lines (one per page), blank lines skipped."""
    text = Path(narration_path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def detect_language(text: str) -> str:
    """Detect language of text, return ISO 639-1 code. Falls back to 'en'."""
    try:
        return langdetect_detect(text)
    except Exception:
        return "en"


def get_tts_voice(lang_code: str) -> str:
    return VOICE_MAP.get(lang_code, VOICE_MAP["default"])


def generate_voiceover(book_dir: str, lines: list) -> list:
    """
    Generate one MP3 per narration line into <book_dir>/audio/.
    Returns sorted list of output MP3 paths.
    Skips lines where the MP3 already exists (caching).
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    audio_dir = Path(book_dir) / "audio"
    audio_dir.mkdir(exist_ok=True)

    lang = detect_language(" ".join(lines))
    voice = get_tts_voice(lang)
    print(f"  Language detected: {lang} → voice: {voice}")

    output_paths = []
    for i, line in enumerate(lines):
        dst = audio_dir / f"page{i+1:02d}.mp3"
        if dst.exists():
            print(f"  Cached: {dst.name}")
        else:
            print(f"  Generating audio: page {i+1}/{len(lines)}")
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=line,
            )
            response.stream_to_file(str(dst))
        output_paths.append(str(dst))
    return output_paths
