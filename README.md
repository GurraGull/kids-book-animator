# Kids Book Animator

Turns children's picture books into animated MP4 videos. Photos of book pages go in, a narrated animated video comes out.

**What it does:**
- AI vision analyzes each page for characters and what they're doing
- Characters get animated (walking, waving, jumping) via Meta Animated Drawings
- Pages without characters get a gentle Ken Burns zoom
- OpenAI TTS narrates the text in the book's language
- Everything assembles into a single MP4 with subtitles, title card, and closing card

---

## Setup

**Requirements:** Python 3.9+, ffmpeg, an OpenAI API key

### 1. Install Python dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Install Meta Animated Drawings (one-time)

```bash
git clone https://github.com/facebookresearch/AnimatedDrawings.git /tmp/AnimatedDrawings
cd /tmp/AnimatedDrawings && pip3 install -e .
cd -
```

### 3. Add your OpenAI API key

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 4. Install ffmpeg (if not already installed)

```bash
brew install ffmpeg   # macOS
```

---

## Usage

### Prepare a book folder

```
books/my-book/
  pages/
    page01.jpg    ← one photo per spread, in order
    page02.jpg
    ...
  narration.txt   ← one line of text per page (what the voice reads)
  music.mp3       ← optional background music
```

**Tips for physical books:**
- Photograph the open book flat on a white table
- One photo per spread (both pages open)
- Good lighting, no shadows

### Run the pipeline

```bash
python3 animate.py books/my-book/ --title "My Book Title"
```

The pipeline will pause and show you the animation plan before rendering. Press `Y` to continue, `n` to abort, or `edit` to manually adjust which pages get animated.

### Output

```
books/my-book/my-book.mp4
```

---

## Test books included

| Book | Pages | Status |
|------|-------|--------|
| Peter Rabbit (Beatrix Potter, 1902) | 12 | Ready — run with command above |

```bash
python3 animate.py books/peter-rabbit/ --title "The Tale of Peter Rabbit"
```

---

## Cost estimate (OpenAI API)

Per book (12 pages, ~2000 characters of narration):
- GPT-4o vision analysis: ~$0.36
- TTS narration: ~$0.03
- **Total: ~$0.40 per book**

Audio is cached — re-running the same book doesn't regenerate audio.

---

## Options

```
python3 animate.py <book_dir> [--title "Title"] [--skip-review] [--dry-run]

  --title        Book title shown on the intro card (default: folder name)
  --skip-review  Skip the animation plan review step
  --dry-run      Run without calling OpenAI APIs — uses placeholder audio and
                 skips character animation. Good for testing the video pipeline.
```
