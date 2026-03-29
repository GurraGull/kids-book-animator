# Children's Book Animator — Alpha Design Spec

**Date:** 2026-03-29
**Status:** Approved
**Scope:** Alpha — personal use, 1–10 books, produces MP4 video files

---

## Overview

A Python CLI pipeline that takes photos of children's book spreads and a narration text file, and produces a single animated MP4 video. Characters on pages are automatically detected and animated using AI vision + Meta Animated Drawings. Static pages get a gentle Ken Burns zoom effect.

**Target books:** 5–10 page picture books for ages 1–4. Simple illustrated characters, one spread per page.

**Single command:**
```
python animate.py books/my-book/
```

---

## Input

Per book, the user prepares a folder with:

| File | Description |
|---|---|
| `pages/page01.jpg` … | Photos of open book spreads, one photo per spread. JPG or PNG. Physical books photographed flat on a table, or digital scans/PDFs. |
| `narration.txt` | One line of text per spread, in order. The voice reads this aloud. Language is auto-detected. |
| `music.mp3` | Optional. Ambient background music, played at low volume under the narration. |

**Example folder:**
```
books/my-book/
  pages/
    page01.jpg
    page02.jpg
    …
  narration.txt
  music.mp3        ← optional
```

---

## Pipeline Steps

### Step 1 — Prepare Images
- If input is a PDF: extract pages as images
- Crop white/table background from spread photos
- Auto-straighten and deskew
- Resize and letterbox to 1920×1080

### Step 2 — Analyze Pages (Vision AI)
- Send each page image to GPT-4 Vision (or Claude claude-sonnet-4-6)
- For each page, detect:
  - Is there an animatable character? (human, animal, creature)
  - What are they doing? (balancing, walking, waving, jumping, sitting, etc.)
- Output: a `plan.json` file with per-page animation decisions
- **Review checkpoint:** print the plan to terminal, ask user to confirm or edit before continuing

**Example plan.json:**
```json
[
  {"page": "page01.jpg", "character": false, "action": null},
  {"page": "page02.jpg", "character": true, "action": "walk", "description": "girl balancing on a log"},
  {"page": "page03.jpg", "character": false, "action": null}
]
```

### Step 3 — Generate Voiceover
- Read `narration.txt`, split into one string per page
- Auto-detect language (langdetect library)
- Send each line to OpenAI TTS API (voice: `nova` — warm, suitable for children's content)
- Save one MP3 per page to `audio/` subfolder
- Skip regenerating if MP3 already exists (caching)

### Step 4 — Animate Characters
For each page marked `character: true` in plan.json:
1. **Segment the character** from the background using `rembg` (pip-installable, no GPU required)
2. **Animate** using Meta Animated Drawings API — pass the cropped character + action label
3. **Composite** the animated character back onto the static background as a video clip
4. Duration of clip = duration of the page's narration audio

For pages with no character:
- Apply Ken Burns effect (slow zoom 100%→115%, randomized pan direction) using FFmpeg
- Duration = duration of narration audio

### Step 5 — Assemble Final Video
Order of clips:
1. **Intro card** (3 seconds): page01 (cover) fades in, book title appears as text overlay, voice reads the title
2. **Page clips** in order, each preceded by a 0.5s crossfade transition
3. **Closing card** (2 seconds): "The End" text fades in on a soft background color sampled from the last page

Subtitle rendering:
- Large white text, bold, centered horizontally, positioned 15% from bottom
- Drop shadow: `0 2px 8px rgba(0,0,0,0.9)`
- Font: system sans-serif (or bundled `Nunito` if available — rounded, child-friendly)
- Text wraps at 80% of frame width

Music (if provided):
- Loop `music.mp3` to match total video duration
- Mix at -18dB under narration

Output: `books/my-book/my-book.mp4`

---

## Review Checkpoint (UX)

After Step 2, the script pauses and prints:

```
=== Animation Plan ===
page01.jpg  →  ken burns (no character detected)
page02.jpg  →  ANIMATE: girl balancing on log → walk
page03.jpg  →  ken burns (no character detected)

Proceed? [Y/n/edit]
```

- `Y` — continue
- `n` — abort
- `edit` — opens `plan.json` in the default editor, re-reads on save

---

## Technology Stack

| Component | Library/Service |
|---|---|
| Image prep | `Pillow`, `opencv-python` |
| Vision AI | OpenAI GPT-4 Vision API |
| Voiceover | OpenAI TTS API (`tts-1`, voice `nova`) |
| Character segmentation | `rembg` |
| Character animation | Meta Animated Drawings (local or API) |
| Video assembly | `FFmpeg` via `ffmpeg-python` |
| Language detection | `langdetect` |

---

## Alpha Test Books (Public Domain)

Three candidate books to use for testing the pipeline — all public domain, free to download:

| # | Book | Author | Why good for alpha |
|---|---|---|---|
| 1 | **The Tale of Peter Rabbit** | Beatrix Potter (1902) | Rabbit hops/runs — perfect for animation. Clean watercolor illustrations, simple backgrounds that segment well. 5–6 key spreads. Ages 2–5. Download from [Project Gutenberg](https://www.gutenberg.org/ebooks/14838) |
| 2 | **A Picture Book for Little Children** | Anonymous (early 1900s) | Directly targeted at ages 1–4. Very simple illustrations. Available on [Internet Archive](https://archive.org/details/picturebookforli00philiala) |
| 3 | **Mother Goose Nursery Rhymes** | Traditional | Short text per page, clear single character per illustration. Many public domain illustrated editions available on [Project Gutenberg](https://www.gutenberg.org/ebooks/bookshelf/22) |

**Recommended first test:** Peter Rabbit — most recognizable, best character for demonstrating animation.

---

## Out of Scope (Alpha)

- OCR / automatic text extraction from page images
- Per-element animation beyond characters (e.g. animating clouds, water)
- Lip sync
- Web UI
- Batch processing multiple books
- Distribution / upload to YouTube

---

## Success Criteria

The alpha is successful when:
1. A 5–10 page picture book folder produces a watchable MP4 in under 10 minutes
2. At least one page has a visibly animated character (walking, waving, or equivalent)
3. Narration audio is correctly synced to each page
4. The video is watchable by a 1–4 year old (smooth, not jarring, subtitles readable)
