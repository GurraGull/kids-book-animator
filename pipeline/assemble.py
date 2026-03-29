import os
import subprocess
from pathlib import Path
import ffmpeg
from pipeline.models import BookPlan


def get_audio_duration(mp3_path: str) -> float:
    """Return duration of an MP3 file in seconds."""
    probe = ffmpeg.probe(mp3_path)
    return float(probe["format"]["duration"])


def build_subtitle_filter(text: str, width: int, height: int) -> str:
    """Build FFmpeg drawtext filter string for large child-friendly subtitles."""
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
    """Create a video clip: Ken Burns zoom on still image + narration audio + subtitle."""
    duration = get_audio_duration(audio_path)
    safe_text = text.replace("'", "\u2019").replace(":", r"\:").replace("\\", "\\\\")

    vf = (
        f"scale=2200:-1,"
        f"zoompan=z='min(zoom+0.0008,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={int(duration * 24)}:s=1920x1080:fps=24,"
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
    """Create 'The End' closing card on a dark background."""
    vf = (
        "drawtext=text='The End':fontsize=96:fontcolor=white:font='Arial Bold'"
        ":x=(w-text_w)/2:y=(h-text_h)/2:shadowx=4:shadowy=4:shadowcolor=black@0.7"
    )
    (
        ffmpeg
        .input("color=c=0x1a0a2e:s=1920x1080:r=24", f="lavfi", t=duration)
        .output(output_path, vf=vf, vcodec="libx264", pix_fmt="yuv420p", an=None)
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
            safe_text = text.replace("'", "\u2019").replace(":", r"\:")
            vf = (
                f"drawtext=text='{safe_text}':fontsize=54:fontcolor=white:font='Arial Bold'"
                f":x=(w-text_w)/2:y=h*0.82:shadowx=3:shadowy=3:shadowcolor=black@0.9"
            )
            (
                ffmpeg
                .input(anim_path)
                .output(ffmpeg.input(audio_path), clip_out, vf=vf,
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
