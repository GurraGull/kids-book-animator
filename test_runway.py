#!/usr/bin/env python3
"""
Single-page Runway animation test.
Tests: vision AI → TTS narration → Runway animation → assembled MP4

Usage:
    python3 test_runway.py

Output: books/test-runway/test-runway.mp4
Cost:   ~$0.25 (one 5s Runway clip) + pennies for OpenAI TTS/vision
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BOOK_DIR = "books/test-runway"
IMAGE_PATH = "books/peter-rabbit/pages/page07.jpg"
NARRATION = "Peter was most dreadfully frightened and rushed all over the garden, for he had forgotten the way back to the gate."
OUTPUT = "books/test-runway/test-runway.mp4"


def check_keys():
    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.environ.get("RUNWAY_API_KEY"):
        missing.append("RUNWAY_API_KEY")
    if missing:
        print(f"❌ Missing in .env: {', '.join(missing)}")
        sys.exit(1)
    print("✅ API keys found")


def step1_vision(image_path):
    print("\n[1/4] Vision AI — detecting characters...")
    from pipeline.analyze import analyze_book
    from pipeline.models import BookPlan, PagePlan
    import base64

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    prompt = (
        "Look at this children's book illustration. "
        "Reply with JSON only: "
        '{"character": true/false, "action": "walk|wave|jump|dance|balance|run|null", "description": "one sentence describing the scene"}'
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        max_tokens=100,
    )
    import json
    raw = resp.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw.strip())
    print(f"   character: {result.get('character')}")
    print(f"   action:    {result.get('action')}")
    print(f"   scene:     {result.get('description', '')[:80]}")
    return result


def step2_normalize(image_path):
    print("\n[2/4] Normalizing image...")
    from pipeline.prepare import normalize_image
    out = "books/test-runway/normalized_page01.jpg"
    normalize_image(image_path, out)
    print(f"   Saved: {out}")
    return out


def step3_tts(narration):
    print("\n[3/4] Generating voiceover (OpenAI TTS)...")
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    audio_out = "books/test-runway/narration.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=narration,
    )
    response.stream_to_file(audio_out)
    print(f"   Saved: {audio_out}")
    return audio_out


def crop_to_character(image_path, tmp_path):
    """Ask GPT-4o for bounding box of main character, crop tight, return cropped path."""
    from openai import OpenAI
    from PIL import Image
    import json, base64
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": (
                "Find the bounding box of the MAIN character (the rabbit) in this image. "
                "Reply with JSON only: {\"x1\": 0.0, \"y1\": 0.0, \"x2\": 1.0, \"y2\": 1.0} "
                "as fractions of image width/height (0.0 to 1.0). Add 10% padding around the character."
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        max_tokens=60,
    )
    raw = resp.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()
    box = json.loads(raw)
    print(f"   Character bbox: {box}")
    img = Image.open(image_path)
    w, h = img.size
    x1 = max(0, int(box["x1"] * w))
    y1 = max(0, int(box["y1"] * h))
    x2 = min(w, int(box["x2"] * w))
    y2 = min(h, int(box["y2"] * h))
    cropped = img.crop((x1, y1, x2, y2))
    cropped.save(tmp_path, "JPEG", quality=95)
    print(f"   Cropped to {x2-x1}x{y2-y1}px → {tmp_path}")
    return tmp_path


def step4_runway(image_path, action, description):
    print(f"\n[4/4] Runway animation ({action})...")
    from pipeline.character import animate_with_runway
    # Crop tight to the character so Runway focuses on it
    cropped = "books/test-runway/character_crop.jpg"
    print(f"   Cropping to main character...")
    image_path = crop_to_character(image_path, cropped)
    out = "books/test-runway/runway_raw.mp4"
    animate_with_runway(image_path, action, description, out, raw_prompt=True)
    return out


def step5_assemble(runway_mp4, audio_path, normalized_img, narration):
    print("\n[5/5] Assembling final clip...")
    from pipeline.assemble import _make_runway_clip
    _make_runway_clip(runway_mp4, audio_path, narration, OUTPUT)
    print(f"\n✅ Done! Output: {OUTPUT}")


if __name__ == "__main__":
    print("=" * 50)
    print("  Runway Single-Page Animation Test")
    print("=" * 50)

    check_keys()

    vision = step1_vision(IMAGE_PATH)
    norm_img = step2_normalize(IMAGE_PATH)
    audio = step3_tts(NARRATION)

    action = vision.get("action") if vision.get("action") and vision.get("action") != "null" else "walk"
    description = vision.get("description") or ""

    if not vision.get("character"):
        print(f"\n⚠️  Vision AI didn't detect a character on this page.")
        print(f"   Runway will still animate it with action: '{action}'")
        print(f"   (You can swap page01.jpg in books/test-runway/pages/ for a better image)")

    description = "Make the girl in the picture move her body."
    action = "walk"
    runway_vid = step4_runway(IMAGE_PATH, action, description)
    step5_assemble(runway_vid, audio, norm_img, NARRATION)

    import subprocess
    subprocess.run(["open", OUTPUT])
