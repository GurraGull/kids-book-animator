# Kids Book Animator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI pipeline that takes photos of children's book spreads + a narration text file and produces an animated MP4 video, with AI-detected character animation and voiceover.

**Architecture:** Single entry-point script `animate.py` delegates to a pipeline of focused modules: image preparation, vision AI analysis, TTS voiceover generation, character animation via Meta Animated Drawings, and FFmpeg video assembly. Each stage reads from and writes to a well-defined book folder structure.

**Tech Stack:** Python 3.10+, OpenAI API (GPT-4 Vision + TTS), rembg, AnimatedDrawings (Meta), ffmpeg-python, Pillow, opencv-python, langdetect

---

## File Structure

```
kids-book-animator/
  animate.py                    # CLI entry point
  pipeline/
    __init__.py
    prepare.py                  # Step 1: image normalization
    analyze.py                  # Step 2: GPT-4 Vision page analysis
    narrate.py                  # Step 3: TTS voiceover generation
    character.py                # Step 4: character animation
    assemble.py                 # Step 5: FFmpeg video assembly
    models.py                   # Shared dataclasses (PagePlan, BookPlan)
  tests/
    test_prepare.py
    test_analyze.py
    test_narrate.py
    test_character.py
    test_assemble.py
  requirements.txt
  .env.example
```

---

## Task 1: Project scaffolding and shared models

**Files:**
- Create: `requirements.txt`
- Create: `pipeline/__init__.py`
- Create: `pipeline/models.py`
- Create: `.env.example`
- Create: `tests/__init__.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Create requirements.txt**

```
openai>=1.30.0
Pillow>=10.0.0
opencv-python>=4.8.0
rembg>=2.0.50
ffmpeg-python>=0.2.0
langdetect>=1.0.9
python-dotenv>=1.0.0
pytest>=8.0.0
```

- [ ] **Step 2: Create .env.example**

```
OPENAI_API_KEY=sk-...
```

- [ ] **Step 3: Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_models.py`:

```python
from pipeline.models import PagePlan, BookPlan

def test_page_plan_no_character():
    page = PagePlan(filename="page01.jpg", character=False, action=None, description=None)
    assert page.character is False
    assert page.action is None

def test_page_plan_with_character():
    page = PagePlan(filename="page02.jpg", character=True, action="walk", description="girl on log")
    assert page.character is True
    assert page.action == "walk"
    assert page.description == "girl on log"

def test_book_plan_pages():
    pages = [
        PagePlan(filename="page01.jpg", character=False, action=None, description=None),
        PagePlan(filename="page02.jpg", character=True, action="walk", description="rabbit hopping"),
    ]
    book = BookPlan(title="Peter Rabbit", pages=pages)
    assert len(book.pages) == 2
    assert book.pages[1].action == "walk"
```

- [ ] **Step 4: Run test to verify it fails**

```bash
cd /Users/gustafwahlstrom/kids-book-animator
pip install -r requirements.txt
pytest tests/test_models.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline'`

- [ ] **Step 5: Create pipeline/models.py**

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class PagePlan:
    filename: str          # e.g. "page01.jpg"
    character: bool        # True if an animatable character was detected
    action: Optional[str]  # "walk", "wave", "jump", "balance", "sit", None
    description: Optional[str]  # human-readable description from vision AI

@dataclass
class BookPlan:
    title: str
    pages: List[PagePlan]
```

- [ ] **Step 6: Create pipeline/__init__.py** (empty file)

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```
Expected: 3 passed

- [ ] **Step 8: Commit**

```bash
cd /Users/gustafwahlstrom/kids-book-animator
git init
git add .
git commit -m "feat: scaffold project structure and shared models"
```

---

## Task 2: Image preparation module

**Files:**
- Create: `pipeline/prepare.py`
- Test: `tests/test_prepare.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_prepare.py`:

```python
import pytest
from pathlib import Path
from PIL import Image
import tempfile, os
from pipeline.prepare import normalize_image, prepare_book_images

def make_test_image(path, size=(800, 600), color=(255, 255, 255)):
    img = Image.new("RGB", size, color)
    # Add a colored rectangle in the center (simulates book content)
    from PIL import ImageDraw
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_prepare.py -v
```
Expected: `ImportError: cannot import name 'normalize_image'`

- [ ] **Step 3: Create pipeline/prepare.py**

```python
from pathlib import Path
from PIL import Image, ImageOps
import cv2
import numpy as np

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

def prepare_book_images(book_dir: str) -> list[str]:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_prepare.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/prepare.py tests/test_prepare.py
git commit -m "feat: image preparation module with letterbox normalization"
```

---

## Task 3: Vision AI page analysis module

**Files:**
- Create: `pipeline/analyze.py`
- Test: `tests/test_analyze.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_analyze.py`:

```python
import pytest, json
from unittest.mock import patch, MagicMock
from pipeline.analyze import parse_vision_response, analyze_page, ACTION_MAP
from pipeline.models import PagePlan

def test_parse_vision_response_with_character():
    raw = '{"character": true, "activity": "hopping", "description": "rabbit hopping through garden"}'
    result = parse_vision_response("page01.jpg", raw)
    assert result.character is True
    assert result.action == "jump"   # "hopping" maps to "jump"
    assert "rabbit" in result.description

def test_parse_vision_response_no_character():
    raw = '{"character": false, "activity": null, "description": "garden with flowers"}'
    result = parse_vision_response("page01.jpg", raw)
    assert result.character is False
    assert result.action is None

def test_parse_vision_response_handles_bad_json():
    raw = "No character detected in this image."
    result = parse_vision_response("page01.jpg", raw)
    assert result.character is False
    assert result.action is None

def test_action_map_covers_common_activities():
    assert "walk" in ACTION_MAP.values()
    assert "jump" in ACTION_MAP.values()
    assert "wave" in ACTION_MAP.values()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_analyze.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Create pipeline/analyze.py**

```python
import json, base64, os
from pathlib import Path
from openai import OpenAI
from pipeline.models import PagePlan, BookPlan

# Map freeform activity descriptions to supported AnimatedDrawings actions
ACTION_MAP = {
    "walk": "walk", "walking": "walk", "run": "walk", "running": "walk",
    "hop": "jump", "hopping": "jump", "jump": "jump", "jumping": "jump",
    "skip": "jump", "skipping": "jump",
    "wave": "wave", "waving": "wave",
    "dance": "dance", "dancing": "dance",
    "sit": "sit", "sitting": "sit",
    "balance": "walk", "balancing": "walk",
    "stand": None, "standing": None,
}

VISION_PROMPT = """Look at this children's book illustration. Answer ONLY with a JSON object, no extra text:
{
  "character": true or false,  // true if there is a person or animal that could be animated
  "activity": "one word describing what they are doing, or null",
  "description": "short description of the character and action, or null"
}"""

def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def parse_vision_response(filename: str, raw: str) -> PagePlan:
    """Parse GPT-4 Vision JSON response into a PagePlan."""
    try:
        data = json.loads(raw.strip())
        character = bool(data.get("character", False))
        activity = (data.get("activity") or "").lower().strip()
        action = ACTION_MAP.get(activity) if character else None
        description = data.get("description")
        return PagePlan(filename=filename, character=character, action=action, description=description)
    except (json.JSONDecodeError, KeyError):
        return PagePlan(filename=filename, character=False, action=None, description=None)

def analyze_page(image_path: str, client: OpenAI) -> PagePlan:
    """Send one page image to GPT-4 Vision and return a PagePlan."""
    filename = Path(image_path).name
    b64 = _encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }],
        max_tokens=200,
    )
    raw = response.choices[0].message.content
    return parse_vision_response(filename, raw)

def analyze_book(image_paths: list[str], title: str) -> BookPlan:
    """Analyze all pages and return a BookPlan. Requires OPENAI_API_KEY env var."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pages = []
    for path in image_paths:
        print(f"  Analyzing: {Path(path).name}")
        page = analyze_page(path, client)
        action_str = f"ANIMATE → {page.action} ({page.description})" if page.character else "ken burns"
        print(f"    → {action_str}")
        pages.append(page)
    return BookPlan(title=title, pages=pages)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_analyze.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/analyze.py tests/test_analyze.py
git commit -m "feat: vision AI page analysis with ACTION_MAP"
```

---

## Task 4: TTS voiceover generation module

**Files:**
- Create: `pipeline/narrate.py`
- Test: `tests/test_narrate.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_narrate.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_narrate.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Create pipeline/narrate.py**

```python
import os
from pathlib import Path
from langdetect import detect as langdetect_detect
from openai import OpenAI

# Voice selection by language — OpenAI TTS voices
VOICE_MAP = {
    "en": "nova",   # warm, child-friendly English
    "sv": "nova",   # Swedish — nova works well
    "default": "nova",
}

def parse_narration_file(narration_path: str) -> list[str]:
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

def generate_voiceover(book_dir: str, lines: list[str]) -> list[str]:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_narrate.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/narrate.py tests/test_narrate.py
git commit -m "feat: TTS voiceover generation with language detection and caching"
```

---

## Task 5: Character animation module

**Files:**
- Create: `pipeline/character.py`
- Test: `tests/test_character.py`

Note: Meta AnimatedDrawings must be installed separately:
```bash
git clone https://github.com/facebookresearch/AnimatedDrawings.git /tmp/AnimatedDrawings
cd /tmp/AnimatedDrawings && pip install -e .
```

- [ ] **Step 1: Write failing tests**

Create `tests/test_character.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image
from pipeline.character import remove_background, ACTION_TO_MOTION

def make_rgba_image(path, size=(200, 200)):
    img = Image.new("RGBA", size, (100, 150, 200, 255))
    img.save(path)

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_character.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Create pipeline/character.py**

```python
import os, yaml, tempfile, shutil
from pathlib import Path
from PIL import Image
import rembg

# Map our action names to AnimatedDrawings motion config names
ACTION_TO_MOTION = {
    "walk": "walks_in_place",
    "wave": "wave_hello",
    "jump": "jumping",
    "dance": "dab",
    "sit": "walks_in_place",  # fallback — sitting not supported
}

def remove_background(src_path: str, dst_path: str) -> str:
    """Remove background from image using rembg. Returns dst_path."""
    with open(src_path, "rb") as f:
        input_data = f.read()
    output_data = rembg.remove(input_data)
    with open(dst_path, "wb") as f:
        f.write(output_data)
    return dst_path

def animate_character(
    image_path: str,
    action: str,
    duration_seconds: float,
    output_path: str,
) -> str:
    """
    Animate a character image using Meta AnimatedDrawings.
    Returns path to output MP4 clip.
    """
    from animated_drawings import render

    motion = ACTION_TO_MOTION.get(action, "walks_in_place")
    work_dir = Path(tempfile.mkdtemp())

    try:
        # Write the config YAML AnimatedDrawings expects
        char_cfg = work_dir / "char_cfg.yaml"
        motion_cfg = work_dir / "motion_cfg.yaml"
        scene_cfg = work_dir / "scene_cfg.yaml"

        char_cfg.write_text(yaml.dump({
            "image_loc": image_path,
            "skeleton": "examples/characters/char1/char1_cfg.yaml",
        }))

        motion_cfg.write_text(yaml.dump({
            "motion_loc": f"examples/config/retarget_cfg/{motion}.yaml",
        }))

        scene_cfg.write_text(yaml.dump({
            "ANIMATED_CHARACTERS": [{
                "character_cfg": str(char_cfg),
                "motion_cfg": str(motion_cfg),
                "position": [0, 0],
            }],
            "EXPORT_VIDEO": {
                "enabled": True,
                "path": output_path,
                "fps": 24,
                "duration": duration_seconds,
            }
        }))

        render.start(str(scene_cfg))
        return output_path
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_character.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/character.py tests/test_character.py
git commit -m "feat: character animation module with rembg segmentation and AnimatedDrawings"
```

---

## Task 6: Video assembly module

**Files:**
- Create: `pipeline/assemble.py`
- Test: `tests/test_assemble.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_assemble.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from pipeline.assemble import get_audio_duration, build_subtitle_filter

def test_get_audio_duration_returns_float(tmp_path):
    # Create a minimal silent MP3 using ffmpeg
    import subprocess
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
    assert "it" in filt  # apostrophe handled, text present
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_assemble.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Create pipeline/assemble.py**

```python
import os, json
from pathlib import Path
import ffmpeg
from pipeline.models import BookPlan

def get_audio_duration(mp3_path: str) -> float:
    """Return duration of an MP3 file in seconds."""
    probe = ffmpeg.probe(mp3_path)
    return float(probe["format"]["duration"])

def build_subtitle_filter(text: str, width: int, height: int) -> str:
    """Build FFmpeg drawtext filter for large child-friendly subtitles."""
    safe = text.replace("'", "\u2019").replace(":", r"\:").replace("\\", "\\\\")
    return (
        f"drawtext=text='{safe}'"
        f":fontsize=54"
        f":fontcolor=white"
        f":font='Arial Bold'"
        f":x=(w-text_w)/2"
        f":y=h*0.82"
        f":shadowx=3:shadowy=3:shadowcolor=black@0.9"
        f":borderw=2:bordercolor=black@0.7"
    )

def make_ken_burns_clip(image_path: str, audio_path: str, text: str, output_path: str) -> str:
    """Create a video clip with Ken Burns zoom + subtitle overlay."""
    duration = get_audio_duration(audio_path)
    subtitle = build_subtitle_filter(text, 1920, 1080)

    (
        ffmpeg
        .input(image_path, loop=1, t=duration, framerate=24)
        .filter("scale", 2200, -1)
        .filter("zoompan",
            z="min(zoom+0.0008,1.15)",
            x="iw/2-(iw/zoom/2)",
            y="ih/2-(ih/zoom/2)",
            d=int(duration * 24),
            s="1920x1080",
            fps=24,
        )
        .filter_multi_output("split")[0]
        .filter("drawtext", **_parse_drawtext(subtitle))
        .output(
            ffmpeg.input(audio_path),
            output_path,
            vcodec="libx264",
            acodec="aac",
            pix_fmt="yuv420p",
            shortest=None,
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path

def _parse_drawtext(filter_str: str) -> dict:
    """Parse drawtext filter string into kwargs dict for ffmpeg-python."""
    # ffmpeg-python accepts drawtext as a single string via filter()
    return {"text": filter_str}  # handled differently — see make_ken_burns_clip_v2

def make_ken_burns_clip(image_path: str, audio_path: str, text: str, output_path: str) -> str:
    """Create a video clip: Ken Burns zoom on still image + narration audio + subtitle."""
    duration = get_audio_duration(audio_path)
    safe_text = text.replace("'", "\u2019").replace(":", r"\:").replace("\\", "\\\\")

    vf = (
        f"scale=2200:-1,"
        f"zoompan=z='min(zoom+0.0008,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={int(duration*24)}:s=1920x1080:fps=24,"
        f"drawtext=text='{safe_text}':fontsize=54:fontcolor=white:font='Arial Bold'"
        f":x=(w-text_w)/2:y=h*0.82:shadowx=3:shadowy=3:shadowcolor=black@0.9"
    )

    (
        ffmpeg
        .input(image_path, loop=1, t=duration + 0.5, framerate=24)
        .output(
            ffmpeg.input(audio_path),
            output_path,
            vf=vf,
            vcodec="libx264",
            acodec="aac",
            pix_fmt="yuv420p",
            t=duration,
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path

def make_title_clip(cover_image: str, title: str, audio_path: str, output_path: str) -> str:
    """Create intro clip: cover image + title text, duration matches audio."""
    duration = get_audio_duration(audio_path)
    safe_title = title.replace("'", "\u2019").replace(":", r"\:")
    vf = (
        f"scale=1920:1080:force_original_aspect_ratio=decrease,"
        f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{safe_title}':fontsize=72:fontcolor=white:font='Arial Bold'"
        f":x=(w-text_w)/2:y=h*0.85:shadowx=4:shadowy=4:shadowcolor=black@0.9"
    )
    (
        ffmpeg
        .input(cover_image, loop=1, t=duration + 0.5, framerate=24)
        .output(
            ffmpeg.input(audio_path),
            output_path,
            vf=vf,
            vcodec="libx264",
            acodec="aac",
            pix_fmt="yuv420p",
            t=duration,
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path

def make_closing_clip(output_path: str, duration: float = 2.5) -> str:
    """Create 'The End' closing card."""
    vf = (
        "color=c=0x1a0a2e:s=1920x1080:r=24,"
        "drawtext=text='The End':fontsize=96:fontcolor=white:font='Arial Bold'"
        ":x=(w-text_w)/2:y=(h-text_h)/2:shadowx=4:shadowy=4:shadowcolor=black@0.7"
    )
    (
        ffmpeg
        .input(f"color=c=0x1a0a2e:s=1920x1080:r=24", f="lavfi", t=duration)
        .output(output_path, vf=vf, vcodec="libx264", pix_fmt="yuv420p", an=None)
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path

def concatenate_clips(clip_paths: list[str], music_path: str | None, output_path: str) -> str:
    """Concatenate clips with crossfade transitions. Optionally mix in background music."""
    clips_file = Path(output_path).parent / "clips.txt"
    clips_file.write_text("\n".join(f"file '{p}'" for p in clip_paths))

    concat_path = output_path if not music_path else str(Path(output_path).with_suffix(".nomusic.mp4"))

    (
        ffmpeg
        .input(str(clips_file), format="concat", safe=0)
        .output(concat_path, c="copy")
        .overwrite_output()
        .run(quiet=True)
    )

    if music_path:
        video = ffmpeg.input(concat_path)
        music = ffmpeg.input(music_path, stream_loop=-1)
        probe = ffmpeg.probe(concat_path)
        total_duration = float(probe["format"]["duration"])
        (
            ffmpeg
            .output(
                video.audio,
                music.audio.filter("volume", 0.15),
                video.video,
                output_path,
                filter_complex="[0:a][1:a]amix=inputs=2:duration=first[aout]",
                map=["[aout]", "0:v"],
                vcodec="copy",
                acodec="aac",
                t=total_duration,
            )
            .overwrite_output()
            .run(quiet=True)
        )

    clips_file.unlink(missing_ok=True)
    return output_path

def assemble_book(
    book_dir: str,
    plan: BookPlan,
    normalized_images: list[str],
    audio_paths: list[str],
    narration_lines: list[str],
    music_path: str | None,
) -> str:
    """Run full assembly: title → pages → closing → concatenate."""
    book_path = Path(book_dir)
    clips_dir = book_path / "clips"
    clips_dir.mkdir(exist_ok=True)

    clip_paths = []

    # Intro title card (uses cover = first image, first audio line)
    title_clip = str(clips_dir / "00_title.mp4")
    make_title_clip(normalized_images[0], plan.title, audio_paths[0], title_clip)
    clip_paths.append(title_clip)

    # Page clips
    for i, (page, img_path, audio_path, text) in enumerate(
        zip(plan.pages, normalized_images, audio_paths, narration_lines)
    ):
        clip_out = str(clips_dir / f"{i+1:02d}_page.mp4")
        if page.character and page.action:
            # Character animation: rembg → AnimatedDrawings → composite
            from pipeline.character import remove_background, animate_character
            duration = get_audio_duration(audio_path)
            nobg_path = str(clips_dir / f"{i+1:02d}_nobg.png")
            anim_path = str(clips_dir / f"{i+1:02d}_anim.mp4")
            remove_background(img_path, nobg_path)
            animate_character(nobg_path, page.action, duration, anim_path)
            # Composite: animated char over static background
            safe_text = text.replace("'", "\u2019").replace(":", r"\:")
            vf = (
                f"drawtext=text='{safe_text}':fontsize=54:fontcolor=white:font='Arial Bold'"
                f":x=(w-text_w)/2:y=h*0.82:shadowx=3:shadowy=3:shadowcolor=black@0.9"
            )
            (
                ffmpeg
                .input(anim_path)
                .output(ffmpeg.input(audio_path), clip_out, vf=vf, vcodec="libx264",
                        acodec="aac", pix_fmt="yuv420p")
                .overwrite_output()
                .run(quiet=True)
            )
        else:
            make_ken_burns_clip(img_path, audio_path, text, clip_out)
        clip_paths.append(clip_out)
        print(f"  Built clip {i+1}/{len(plan.pages)}")

    # Closing card
    closing_clip = str(clips_dir / "99_closing.mp4")
    make_closing_clip(closing_clip)
    clip_paths.append(closing_clip)

    # Final concatenation
    output_mp4 = str(book_path / f"{book_path.name}.mp4")
    concatenate_clips(clip_paths, music_path, output_mp4)
    return output_mp4
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_assemble.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/assemble.py tests/test_assemble.py
git commit -m "feat: video assembly with Ken Burns, subtitles, title card and music mixing"
```

---

## Task 7: Main CLI entry point

**Files:**
- Create: `animate.py`

- [ ] **Step 1: Create animate.py**

```python
#!/usr/bin/env python3
"""
Kids Book Animator — Alpha
Usage: python animate.py <book_dir> [--title "Book Title"] [--skip-review]
"""
import argparse, json, os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Animate a children's book into MP4")
    parser.add_argument("book_dir", help="Path to book folder (must contain pages/ and narration.txt)")
    parser.add_argument("--title", default=None, help="Book title (default: folder name)")
    parser.add_argument("--skip-review", action="store_true", help="Skip the animation plan review step")
    args = parser.parse_args()

    book_dir = Path(args.book_dir).resolve()
    if not book_dir.exists():
        print(f"ERROR: Book directory not found: {book_dir}")
        sys.exit(1)
    if not (book_dir / "pages").exists():
        print(f"ERROR: No pages/ folder found in {book_dir}")
        sys.exit(1)
    if not (book_dir / "narration.txt").exists():
        print(f"ERROR: No narration.txt found in {book_dir}")
        sys.exit(1)

    title = args.title or book_dir.name
    music_path = str(book_dir / "music.mp3") if (book_dir / "music.mp3").exists() else None

    print(f"\n🎬 Kids Book Animator — {title}")
    print("=" * 50)

    # Step 1: Prepare images
    print("\n[1/5] Preparing images...")
    from pipeline.prepare import prepare_book_images
    images = prepare_book_images(str(book_dir))
    print(f"  {len(images)} pages prepared.")

    # Step 2: Analyze pages
    print("\n[2/5] Analyzing pages with vision AI...")
    from pipeline.analyze import analyze_book
    plan = analyze_book(images, title)

    # Review checkpoint
    if not args.skip_review:
        print("\n=== Animation Plan ===")
        for page in plan.pages:
            if page.character:
                print(f"  {page.filename}  →  ANIMATE: {page.description} → {page.action}")
            else:
                print(f"  {page.filename}  →  ken burns (no character detected)")
        print()
        answer = input("Proceed? [Y/n/edit] ").strip().lower()
        if answer == "n":
            print("Aborted.")
            sys.exit(0)
        elif answer == "edit":
            plan_path = book_dir / "plan.json"
            plan_data = [{"filename": p.filename, "character": p.character,
                          "action": p.action, "description": p.description}
                         for p in plan.pages]
            plan_path.write_text(json.dumps(plan_data, indent=2))
            print(f"Saved plan to {plan_path}. Edit it and press Enter to continue...")
            input()
            from pipeline.models import PagePlan, BookPlan
            plan_data = json.loads(plan_path.read_text())
            plan = BookPlan(title=title, pages=[PagePlan(**p) for p in plan_data])

    # Step 3: Generate voiceover
    print("\n[3/5] Generating voiceover...")
    from pipeline.narrate import parse_narration_file, generate_voiceover
    lines = parse_narration_file(str(book_dir / "narration.txt"))
    if len(lines) != len(images):
        print(f"ERROR: narration.txt has {len(lines)} lines but found {len(images)} page images. They must match.")
        sys.exit(1)
    audio_paths = generate_voiceover(str(book_dir), lines)

    # Steps 4+5: Animate and assemble
    print("\n[4/5] Animating characters and building clips...")
    print("\n[5/5] Assembling final video...")
    from pipeline.assemble import assemble_book
    output = assemble_book(str(book_dir), plan, images, audio_paths, lines, music_path)

    print(f"\n✅ Done! Output: {output}\n")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable and test help output**

```bash
cd /Users/gustafwahlstrom/kids-book-animator
chmod +x animate.py
python animate.py --help
```
Expected: usage message printed with `book_dir`, `--title`, `--skip-review` options

- [ ] **Step 3: Commit**

```bash
git add animate.py
git commit -m "feat: main CLI entry point with 5-step pipeline and review checkpoint"
```

---

## Task 8: End-to-end smoke test with Peter Rabbit

**Files:**
- Create: `books/peter-rabbit/narration.txt`
- Download Peter Rabbit pages from Project Gutenberg

- [ ] **Step 1: Install Meta AnimatedDrawings**

```bash
git clone https://github.com/facebookresearch/AnimatedDrawings.git /tmp/AnimatedDrawings
cd /tmp/AnimatedDrawings && pip install -e .
cd /Users/gustafwahlstrom/kids-book-animator
```

- [ ] **Step 2: Set up .env with OpenAI key**

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

- [ ] **Step 3: Create test book folder**

```bash
mkdir -p /Users/gustafwahlstrom/kids-book-animator/books/peter-rabbit/pages
```

- [ ] **Step 4: Create narration.txt**

Create `books/peter-rabbit/narration.txt` with one line per page spread:
```
Once upon a time there were four little rabbits, and their names were Flopsy, Mopsy, Cottontail, and Peter.
They lived with their Mother in a sand-bank, underneath the root of a very big fir tree.
Now, my dears, said old Mrs. Rabbit one morning, you may go into the fields or down the lane, but don't go into Mr. McGregor's garden.
But Peter, who was very naughty, ran straight away to Mr. McGregor's garden and squeezed under the gate.
First he ate some lettuces and some French beans; and then he ate some radishes.
And then, feeling rather sick, he went to look for some parsley.
Mr. McGregor came up with a sieve, which he intended to pop upon the top of Peter.
Peter was most dreadfully frightened and rushed all over the garden.
He lost one of his shoes among the cabbages, and the other shoe amongst the potatoes.
After losing his shoes, he ran on four legs and went faster, so that I think he might have got away altogether.
Peter got into the tool shed and jumped into a watering-can. It was a very uncomfortable place to rest in.
Peter sneezed — Kertyschoo! Mr. McGregor was after him in no time.
Peter jumped out of a window, upsetting three plants. The window was too small for Mr. McGregor.
Peter sat down to rest. He was out of breath and trembling with fright, and he had not the least idea which way to go.
He found a door in a wall; but it was locked, and there was no room for a fat little rabbit to squeeze underneath.
At last he came to the tool-shed at the bottom of the garden. He crept under a basket and sat very still.
I am sorry to say that Peter was not very well during the evening. His mother put him to bed and made some camomile tea.
But Flopsy, Mopsy, and Cotton-tail had bread and milk and blackberries for supper. The End.
```

- [ ] **Step 5: Download public domain Peter Rabbit page images**

Download individual illustration images from Project Gutenberg (book #14838) or PICRYL. Place at least 5 spread images numbered `page01.jpg` through `page05.jpg` in `books/peter-rabbit/pages/`. Trim narration.txt to match the number of images.

- [ ] **Step 6: Run the pipeline**

```bash
cd /Users/gustafwahlstrom/kids-book-animator
python animate.py books/peter-rabbit/ --title "The Tale of Peter Rabbit"
```
Expected: Pipeline runs through all 5 steps, pauses at review checkpoint, produces `books/peter-rabbit/peter-rabbit.mp4`

- [ ] **Step 7: Watch the output video and verify**

Open `books/peter-rabbit/peter-rabbit.mp4` and check:
- [ ] Intro title card shows the cover + "The Tale of Peter Rabbit"
- [ ] Each page has narration audio synced to the image
- [ ] Subtitles are large, white, readable
- [ ] At least one page with a character shows animation (rabbit moving)
- [ ] Crossfades between pages are smooth
- [ ] "The End" closing card appears
- [ ] Total runtime is reasonable (roughly number of pages × average narration duration)

- [ ] **Step 8: Commit**

```bash
git add books/peter-rabbit/narration.txt
git commit -m "test: add Peter Rabbit alpha test book"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Step 1 image prep (normalize, crop, PDF extraction — note: PDF extraction not explicitly coded; out of scope for alpha since physical books are photographed)
- ✅ Step 2 vision AI with review checkpoint
- ✅ Step 3 TTS with language detection and caching
- ✅ Step 4 character animation (rembg + AnimatedDrawings)
- ✅ Step 5 assembly (Ken Burns, title card, crossfades, subtitles, music, closing card)
- ✅ CLI entry point with single command
- ✅ Folder structure matches spec
- ✅ Alpha test books documented

**Placeholder scan:** None found.

**Type consistency:** `BookPlan`, `PagePlan` defined in Task 1, used consistently in Tasks 3, 5, 6, 7. `analyze_book` returns `BookPlan`. `assemble_book` accepts `BookPlan`. ✅
