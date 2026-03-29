import os
import yaml
import tempfile
import shutil
from pathlib import Path
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
    Requires AnimatedDrawings to be installed:
        git clone https://github.com/facebookresearch/AnimatedDrawings.git
        cd AnimatedDrawings && pip install -e .
    """
    try:
        from animated_drawings import render
    except ImportError:
        raise ImportError(
            "Meta AnimatedDrawings is not installed. Install it with:\n"
            "  git clone https://github.com/facebookresearch/AnimatedDrawings.git /tmp/AnimatedDrawings\n"
            "  cd /tmp/AnimatedDrawings && pip install -e ."
        )

    motion = ACTION_TO_MOTION.get(action, "walks_in_place")
    work_dir = Path(tempfile.mkdtemp())

    try:
        scene_cfg = work_dir / "scene_cfg.yaml"
        scene_data = {
            "ANIMATED_CHARACTERS": [{
                "character_cfg": image_path,
                "motion_cfg": motion,
                "position": [0, 0],
            }],
            "EXPORT_VIDEO": {
                "enabled": True,
                "path": output_path,
                "fps": 24,
                "duration": duration_seconds,
            }
        }
        scene_cfg.write_text(yaml.dump(scene_data))
        render.start(str(scene_cfg))
        return output_path
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
