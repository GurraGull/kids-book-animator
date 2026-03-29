#!/usr/bin/env python3
"""
Kids Book Animator — Alpha
Usage: python animate.py <book_dir> [--title "Book Title"] [--skip-review] [--dry-run]
"""
import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _generate_silent_audio(book_dir: str, count: int) -> list:
    """Generate short silent MP3 files for dry-run mode."""
    import subprocess
    audio_dir = Path(book_dir) / "audio"
    audio_dir.mkdir(exist_ok=True)
    paths = []
    for i in range(count):
        dst = audio_dir / f"page{i+1:02d}.mp3"
        if not dst.exists():
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
                "-t", "4", "-q:a", "9", "-acodec", "libmp3lame", str(dst), "-y"
            ], check=True, capture_output=True)
        paths.append(str(dst))
    return paths


def main():
    parser = argparse.ArgumentParser(description="Animate a children's book into MP4")
    parser.add_argument("book_dir", help="Path to book folder (must contain pages/ and narration.txt)")
    parser.add_argument("--title", default=None, help="Book title (default: folder name)")
    parser.add_argument("--skip-review", action="store_true", help="Skip the animation plan review step")
    parser.add_argument("--dry-run", action="store_true", help="Skip OpenAI API calls — use silent audio and no character animation. Tests the video pipeline for free.")
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
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
        print("       Or run with --dry-run to test the pipeline without API calls.")
        sys.exit(1)

    title = args.title or book_dir.name
    music_path = str(book_dir / "music.mp3") if (book_dir / "music.mp3").exists() else None

    print(f"\n🎬 Kids Book Animator — {title}")
    if args.dry_run:
        print("   [DRY RUN — no API calls, silent audio]")
    print("=" * 50)

    # Step 1: Prepare images
    print("\n[1/5] Preparing images...")
    from pipeline.prepare import prepare_book_images
    images = prepare_book_images(str(book_dir))
    print(f"  {len(images)} pages prepared.")

    # Step 2: Analyze pages
    print("\n[2/5] Analyzing pages with vision AI...")
    if args.dry_run:
        from pipeline.models import PagePlan, BookPlan
        plan = BookPlan(title=title, pages=[
            PagePlan(filename=Path(p).name, character=False, action=None, description=None)
            for p in images
        ])
        print("  [dry-run] Skipped — all pages set to ken burns.")
    else:
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
            plan_data = [
                {
                    "filename": p.filename,
                    "character": p.character,
                    "action": p.action,
                    "description": p.description,
                }
                for p in plan.pages
            ]
            plan_path.write_text(json.dumps(plan_data, indent=2))
            print(f"Saved plan to {plan_path}. Edit it and press Enter to continue...")
            input()
            from pipeline.models import PagePlan, BookPlan
            plan_data = json.loads(plan_path.read_text())
            plan = BookPlan(title=title, pages=[PagePlan(**p) for p in plan_data])

    # Step 3: Generate voiceover
    print("\n[3/5] Generating voiceover...")
    from pipeline.narrate import parse_narration_file
    lines = parse_narration_file(str(book_dir / "narration.txt"))
    if len(lines) != len(images):
        print(
            f"ERROR: narration.txt has {len(lines)} lines but found {len(images)} page images. "
            "They must match exactly."
        )
        sys.exit(1)

    if args.dry_run:
        audio_paths = _generate_silent_audio(str(book_dir), len(lines))
        print(f"  [dry-run] Generated {len(audio_paths)} silent audio clips.")
    else:
        from pipeline.narrate import generate_voiceover
        audio_paths = generate_voiceover(str(book_dir), lines)

    # Steps 4+5: Animate and assemble
    print("\n[4/5] Animating characters and building clips...")
    print("[5/5] Assembling final video...")
    from pipeline.assemble import assemble_book
    output = assemble_book(str(book_dir), plan, images, audio_paths, lines, music_path)

    print(f"\n✅ Done! Output: {output}\n")


if __name__ == "__main__":
    main()
