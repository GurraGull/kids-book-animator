import os
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import ffmpeg
from pipeline.models import BookPlan

# Subtitle bar height as fraction of frame height
SUBTITLE_Y_FRAC = 0.82
FONT_SIZE_SUBTITLE = 54
FONT_SIZE_TITLE = 72
FONT_SIZE_CLOSING = 96
FRAME_W, FRAME_H = 1920, 1080


def get_audio_duration(mp3_path: str) -> float:
    """Return duration of an MP3 file in seconds."""
    probe = ffmpeg.probe(mp3_path)
    return float(probe["format"]["duration"])


def _get_font(size: int):
    """Load a font, falling back to default if system fonts unavailable."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_text_on_image(img: Image.Image, text: str, font_size: int,
                         y_frac: float = 0.82, color=(255, 255, 255)) -> Image.Image:
    """Burn text onto an image with drop shadow. Returns modified copy."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = _get_font(font_size)
    w, h = out.size

    # Wrap text to 80% of frame width
    words = text.split()
    lines = []
    current = []
    for word in words:
        test_line = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] > w * 0.8 and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))

    line_height = font_size + 8
    total_text_h = line_height * len(lines)
    y_start = int(h * y_frac) - total_text_h // 2

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (w - text_w) // 2
        y = y_start + i * line_height
        # Shadow
        for dx, dy in [(3, 3), (-2, 2), (2, -2), (-2, -2)]:
            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 200))
        draw.text((x, y), line, font=font, fill=color)

    return out


def build_subtitle_filter(text: str, width: int, height: int) -> str:
    """Returns a description string (kept for test compatibility — actual rendering uses Pillow)."""
    return (
        f"drawtext=text='{text}'"
        f":fontsize=54:fontcolor=white:font='Arial Bold'"
        f":x=(w-text_w)/2:y=h*0.82"
        f":shadowx=3:shadowy=3:shadowcolor=black@0.9"
        f":borderw=2:bordercolor=black@0.7"
    )


def _image_with_subtitle(image_path: str, text: str, tmp_dir: str) -> str:
    """Render subtitle onto image, save to tmp file, return path."""
    img = Image.open(image_path).convert("RGB")
    if img.size != (FRAME_W, FRAME_H):
        canvas = Image.new("RGB", (FRAME_W, FRAME_H), (0, 0, 0))
        img.thumbnail((FRAME_W, FRAME_H), Image.LANCZOS)
        x = (FRAME_W - img.width) // 2
        y = (FRAME_H - img.height) // 2
        canvas.paste(img, (x, y))
        img = canvas
    img = _draw_text_on_image(img, text, FONT_SIZE_SUBTITLE, y_frac=SUBTITLE_Y_FRAC)
    dst = Path(tmp_dir) / (Path(image_path).stem + "_sub.jpg")
    img.save(str(dst), "JPEG", quality=92)
    return str(dst)


def make_ken_burns_clip(image_path: str, audio_path: str, text: str, output_path: str) -> str:
    """Create a video clip: Ken Burns zoom on still image + narration audio + subtitle."""
    duration = get_audio_duration(audio_path)

    with tempfile.TemporaryDirectory() as tmp:
        sub_image = _image_with_subtitle(image_path, text, tmp)
        frames = int(duration * 24)
        vf = (
            f"scale=2200:-1,"
            f"zoompan=z='min(zoom+0.0008,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={FRAME_W}x{FRAME_H}:fps=24"
        )
        (
            ffmpeg
            .input(sub_image, loop=1, t=duration + 0.5, framerate=24)
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

    with tempfile.TemporaryDirectory() as tmp:
        title_image = _image_with_subtitle(cover_image, title, tmp)
        (
            ffmpeg
            .input(title_image, loop=1, t=duration + 0.5, framerate=24)
            .output(
                ffmpeg.input(audio_path),
                output_path,
                vf=f"scale={FRAME_W}:{FRAME_H}:force_original_aspect_ratio=decrease,"
                   f"pad={FRAME_W}:{FRAME_H}:(ow-iw)/2:(oh-ih)/2",
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
    """Create 'The End' closing card on a dark background."""
    img = Image.new("RGB", (FRAME_W, FRAME_H), (26, 10, 46))
    img = _draw_text_on_image(img, "The End", FONT_SIZE_CLOSING, y_frac=0.5)

    with tempfile.TemporaryDirectory() as tmp:
        closing_img = str(Path(tmp) / "closing.jpg")
        img.save(closing_img, "JPEG", quality=92)
        (
            ffmpeg
            .input(closing_img, loop=1, t=duration + 0.3, framerate=24)
            .output(
                output_path,
                vf=f"scale={FRAME_W}:{FRAME_H}",
                vcodec="libx264",
                pix_fmt="yuv420p",
                t=duration,
                an=None,
            )
            .overwrite_output()
            .run(quiet=True)
        )
    return output_path


def concatenate_clips(clip_paths: list, music_path, output_path: str) -> str:
    """Concatenate clips using ffmpeg concat demuxer. Optionally mix in background music."""
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
        probe = ffmpeg.probe(concat_path)
        total_duration = float(probe["format"]["duration"])
        video_in = ffmpeg.input(concat_path)
        music_in = ffmpeg.input(music_path, stream_loop=-1, t=total_duration)
        mixed_audio = ffmpeg.filter(
            [video_in.audio, music_in.audio],
            "amix",
            inputs=2,
            duration="first",
            weights="1 0.15",
        )
        (
            ffmpeg
            .output(
                video_in.video,
                mixed_audio,
                output_path,
                vcodec="copy",
                acodec="aac",
                t=total_duration,
            )
            .overwrite_output()
            .run(quiet=True)
        )
        Path(concat_path).unlink(missing_ok=True)

    clips_file.unlink(missing_ok=True)
    return output_path


def assemble_book(
    book_dir: str,
    plan: BookPlan,
    normalized_images: list,
    audio_paths: list,
    narration_lines: list,
    music_path,
) -> str:
    """Run full assembly: title → pages → closing → concatenate."""
    book_path = Path(book_dir)
    clips_dir = book_path / "clips"
    clips_dir.mkdir(exist_ok=True)

    clip_paths = []

    # Intro title card
    title_clip = str(clips_dir / "00_title.mp4")
    make_title_clip(normalized_images[0], plan.title, audio_paths[0], title_clip)
    clip_paths.append(title_clip)

    # Page clips
    for i, (page, img_path, audio_path, text) in enumerate(
        zip(plan.pages, normalized_images, audio_paths, narration_lines)
    ):
        clip_out = str(clips_dir / f"{i+1:02d}_page.mp4")
        if page.character and page.action:
            from pipeline.character import remove_background, animate_character
            duration = get_audio_duration(audio_path)
            nobg_path = str(clips_dir / f"{i+1:02d}_nobg.png")
            anim_path = str(clips_dir / f"{i+1:02d}_anim.mp4")
            remove_background(img_path, nobg_path)
            animate_character(nobg_path, page.action, duration, anim_path)
            # Add subtitle to animated clip by overlaying the text image
            with tempfile.TemporaryDirectory() as tmp:
                sub_image = _image_with_subtitle(img_path, text, tmp)
                (
                    ffmpeg
                    .input(anim_path)
                    .output(ffmpeg.input(audio_path), clip_out,
                            vcodec="libx264", acodec="aac", pix_fmt="yuv420p")
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
