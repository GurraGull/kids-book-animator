# Kids Book Animator — Claude Code Context

## What this project does
Python CLI pipeline that turns children's book pages into animated MP4 videos.
Photos/scans → AI character detection → Runway animation or Ken Burns → TTS narration → assembled video.

## Run the pipeline
```bash
python3 animate.py books/<book-name>/ --title "Book Title"
python3 animate.py books/<book-name>/ --dry-run --skip-review   # no API calls, silent audio
python3 test_runway.py                                           # single-page Runway test
```

## Project structure
- `animate.py` — CLI entry point (5-step pipeline)
- `pipeline/prepare.py` — normalize images, blurred background, PDF extraction
- `pipeline/analyze.py` — GPT-4o Vision: detect characters + action per page
- `pipeline/narrate.py` — OpenAI TTS (voice: nova), one MP3 per page
- `pipeline/character.py` — Runway Gen-4 Turbo image-to-video animation
- `pipeline/assemble.py` — ffmpeg: Ken Burns or Runway clips → crossfade → MP4
- `pipeline/models.py` — PagePlan / BookPlan dataclasses
- `books/<name>/pages/` — input images (JPG/PNG) or single book.pdf
- `books/<name>/narration.txt` — one line per page

## API keys (in .env)
- `OPENAI_API_KEY` — GPT-4o Vision + TTS narration
- `RUNWAY_API_KEY` — Gen-4 Turbo animation (~25 credits / page = $0.25)

## Animation logic
- Page has character detected → Runway Gen-4 Turbo (raw_prompt=True, short simple prompts work best e.g. "Make the girl move her body")
- No character → Ken Burns smooth zoom
- Both cases: blurred page image as background, subtitle bar at bottom, 1.2s pause after narration

## What we learned about Runway
- Simple natural language prompts work best: "Make the character move their body"
- Send the original page image (not normalized) to Runway
- Character must be large in frame — small characters in busy scenes don't animate well
- gen4_turbo = 5 credits/second, minimum 5s = 25 credits = $0.25/clip

## Test books
- `books/peter-rabbit/` — 12 pages, Beatrix Potter (public domain)
- `books/mother-goose/` — 8 pages, nursery rhymes (public domain)
- `books/test-runway/` — single page Runway test setup

## Stack
Python 3.9, OpenAI SDK, RunwayML SDK, ffmpeg-python, Pillow, rembg, langdetect, pdf2image
