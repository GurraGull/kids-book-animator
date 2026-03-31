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
    """Burn text onto an image with a semi-transparent bar and drop shadow."""
    out = img.copy().convert("RGBA")
    font = _get_font(font_size)
    w, h = out.size

    # Wrap text to 80% of frame width
    dummy_draw = ImageDraw.Draw(out)
    words = text.split()
    lines = []
    current = []
    for word in words:
        test_line = " ".join(current + [word])
        bbox = dummy_draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] > w * 0.8 and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))

    line_height = font_size + 10
    pad = 18
    total_text_h = line_height * len(lines)
    bar_h = total_text_h + pad * 2
    bar_top = int(h * y_frac) - bar_h // 2

    # Draw semi-transparent dark bar
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    bar_draw = ImageDraw.Draw(overlay)
    bar_draw.rectangle([(0, bar_top), (w, bar_top + bar_h)], fill=(0, 0, 0, 160))
    out = Image.alpha_composite(out, overlay)

    # Draw text on top
    draw = ImageDraw.Draw(out)
    y_start = bar_top + pad
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (w - text_w) // 2
        y = y_start + i * line_height
        # Drop shadow
        for dx, dy in [(3, 3), (-2, 2), (2, -2)]:
            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 220))
        draw.text((x, y), line, font=font, fill=color)

    return out.convert("RGB")


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


def _subtitle_overlay_png(text: str, tmp_dir: str) -> str:
    """Create a transparent PNG of just the subtitle bar — for overlaying on video."""
    canvas = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
    font = _get_font(FONT_SIZE_SUBTITLE)
    w, h = canvas.size

    dummy_draw = ImageDraw.Draw(canvas)
    words = text.split()
    lines = []
    current = []
    for word in words:
        test_line = " ".join(current + [word])
        bbox = dummy_draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] > w * 0.8 and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))

    line_height = FONT_SIZE_SUBTITLE + 10
    pad = 18
    total_text_h = line_height * len(lines)
    bar_h = total_text_h + pad * 2
    bar_top = int(h * SUBTITLE_Y_FRAC) - bar_h // 2

    bar_draw = ImageDraw.Draw(canvas)
    bar_draw.rectangle([(0, bar_top), (w, bar_top + bar_h)], fill=(0, 0, 0, 160))

    draw = ImageDraw.Draw(canvas)
    y_start = bar_top + pad
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (w - text_w) // 2
        y = y_start + i * line_height
        for dx, dy in [(3, 3), (-2, 2), (2, -2)]:
            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 220))
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))

    dst = Path(tmp_dir) / "subtitle_overlay.png"
    canvas.save(str(dst), "PNG")
    return str(dst)


def make_ken_burns_clip(image_path: str, audio_path: str, text: str, output_path: str) -> str:
    """Create a video clip: Ken Burns zoom on still image + narration audio + subtitle."""
    audio_duration = get_audio_duration(audio_path)
    PAUSE = 1.2  # seconds of still image after narration ends, before next page
    duration = audio_duration + PAUSE

    with tempfile.TemporaryDirectory() as tmp:
        sub_image = _image_with_subtitle(image_path, text, tmp)
        frames = int(duration * 24)
        # Smooth Ken Burns: very gentle zoom-in, centered, minimal jitter
        # scale to slightly larger than output, then zoompan with slow linear zoom
        vf = (
            f"scale={FRAME_W * 2}:{FRAME_H * 2}:flags=lanczos,"
            f"zoompan=z='1.0+0.0003*on':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2'"
            f":d={frames}:s={FRAME_W}x{FRAME_H}:fps=24"
        )
        # Pad audio with silence so clip total duration = audio + pause
        audio_in = ffmpeg.input(audio_path)
        silence = ffmpeg.input("anullsrc=r=44100:cl=stereo", f="lavfi", t=PAUSE)
        padded_audio = ffmpeg.filter([audio_in, silence], "concat", n=2, v=0, a=1)
        (
            ffmpeg
            .input(sub_image, loop=1, t=duration + 0.5, framerate=24)
            .output(
                padded_audio,
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
    """Create 'The End' closing card on a dark background, with silent audio."""
    img = Image.new("RGB", (FRAME_W, FRAME_H), (26, 10, 46))
    img = _draw_text_on_image(img, "The End", FONT_SIZE_CLOSING, y_frac=0.5)

    with tempfile.TemporaryDirectory() as tmp:
        closing_img = str(Path(tmp) / "closing.jpg")
        img.save(closing_img, "JPEG", quality=92)
        video = ffmpeg.input(closing_img, loop=1, t=duration + 0.3, framerate=24)
        silent = ffmpeg.input("anullsrc=r=44100:cl=stereo", f="lavfi", t=duration)
        (
            ffmpeg
            .output(
                video, silent,
                output_path,
                vf=f"scale={FRAME_W}:{FRAME_H}",
                vcodec="libx264",
                acodec="aac",
                pix_fmt="yuv420p",
                t=duration,
            )
            .overwrite_output()
            .run(quiet=True)
        )
    return output_path


def _apply_crossfades(clip_paths: list, output_path: str, fade_duration: float = 0.5) -> str:
    """
    Join clips with xfade crossfade transitions.
    Each clip must have a video stream. Audio is handled separately (merged after).
    """
    if len(clip_paths) == 1:
        # Nothing to crossfade — just copy
        (
            ffmpeg.input(clip_paths[0])
            .output(output_path, c="copy")
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path

    # Get durations to calculate xfade offsets
    durations = [float(ffmpeg.probe(p)["format"]["duration"]) for p in clip_paths]

    # Build complex filter: chain xfade across all clips
    inputs = [ffmpeg.input(p) for p in clip_paths]
    video_streams = [inp.video for inp in inputs]
    audio_streams = [inp.audio for inp in inputs]

    # Chain video xfades
    offset = durations[0] - fade_duration
    v = video_streams[0]
    for i in range(1, len(clip_paths)):
        v = ffmpeg.filter([v, video_streams[i]], "xfade",
                          transition="fade",
                          duration=fade_duration,
                          offset=max(0.0, offset))
        offset += durations[i] - fade_duration

    # Concatenate audio streams (no crossfade on audio — keeps sync simple)
    audio_concat = ffmpeg.filter(audio_streams, "concat", n=len(audio_streams), v=0, a=1)

    (
        ffmpeg
        .output(v, audio_concat, output_path, vcodec="libx264", acodec="aac", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path


def concatenate_clips(clip_paths: list, music_path, output_path: str) -> str:
    """Concatenate clips with crossfade transitions. Optionally mix in background music."""
    concat_path = output_path if not music_path else str(Path(output_path).with_suffix(".nomusic.mp4"))

    print("  Applying crossfade transitions...")
    _apply_crossfades(clip_paths, concat_path)

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

    return output_path


def _make_runway_clip(runway_mp4: str, audio_path: str, text: str, output_path: str) -> str:
    """
    Combine a Runway-generated video with narration audio + subtitle overlay.
    Loops the Runway video if it's shorter than the audio + pause.
    """
    PAUSE = 1.2
    audio_dur = get_audio_duration(audio_path)
    total_dur = audio_dur + PAUSE

    # Build padded audio: narration + 1.2s silence
    silence = ffmpeg.input("anullsrc=r=44100:cl=stereo", f="lavfi", t=PAUSE)
    padded_audio = ffmpeg.filter([ffmpeg.input(audio_path), silence], "concat", n=2, v=0, a=1)

    with tempfile.TemporaryDirectory() as tmp:
        # Create subtitle overlay PNG (transparent background, text bar at bottom)
        sub_png = _subtitle_overlay_png(text, tmp)

        # Runway video: loop + scale to 1920x1080
        base_video = (
            ffmpeg.input(runway_mp4, stream_loop=-1, t=total_dur)
            .video
            .filter("scale", FRAME_W, FRAME_H)
        )
        # Subtitle overlay: static PNG looped as video
        sub_video = ffmpeg.input(sub_png, loop=1, t=total_dur, framerate=24).video

        # Composite: Runway video underneath, subtitle PNG on top
        composited = ffmpeg.filter([base_video, sub_video], "overlay", x=0, y=0)

        (
            ffmpeg
            .output(composited, padded_audio, output_path,
                    vcodec="libx264", acodec="aac", pix_fmt="yuv420p", t=total_dur)
            .overwrite_output()
            .run(quiet=True)
        )
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
        runway_key = os.environ.get("RUNWAY_API_KEY")
        if page.character and page.action and runway_key:
            from pipeline.character import animate_with_runway
            print(f"  Animating page {i+1} with Runway ({page.action})...")
            raw_anim = str(clips_dir / f"{i+1:02d}_runway_raw.mp4")
            animate_with_runway(img_path, page.action, page.description or "", raw_anim)
            _make_runway_clip(raw_anim, audio_path, text, clip_out)
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
