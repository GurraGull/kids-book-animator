import os
import time
import base64
import tempfile
import urllib.request
from pathlib import Path

# Motion prompt templates — these get combined with the scene description
# Runway responds best to specific, cinematic language about what moves and how
ACTION_TO_PROMPT = {
    "walk":    "Make the character in the picture walk.",
    "wave":    "Make the character in the picture wave their hand.",
    "jump":    "Make the character in the picture jump.",
    "dance":   "Make the character in the picture dance.",
    "sit":     "Make the character in the picture move their body gently.",
    "balance": "Make the character in the picture move their body and arms for balance.",
    "run":     "Make the character in the picture run.",
}

RUNWAY_MODEL = "gen4_turbo"
VIDEO_DURATION = 5  # seconds — gen4_turbo minimum is 5s


def _image_to_data_uri(image_path: str) -> str:
    """Convert a local image to a base64 data URI for Runway API."""
    with open(image_path, "rb") as f:
        data = f.read()
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def animate_with_runway(
    image_path: str,
    action: str,
    description: str,
    output_path: str,
    raw_prompt: bool = False,
) -> str:
    """
    Animate a book page image using Runway Gen-4 Turbo image-to-video API.
    Returns path to downloaded MP4 clip.

    Requires RUNWAY_API_KEY in environment.
    Cost: ~$0.25 per 5-second clip (gen4_turbo).
    """
    try:
        from runwayml import RunwayML
    except ImportError:
        raise ImportError(
            "runwayml is not installed. Install it with:\n"
            "  pip3 install runwayml"
        )

    api_key = os.environ.get("RUNWAY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "RUNWAY_API_KEY not set. Add it to your .env file:\n"
            "  RUNWAY_API_KEY=your_key_here"
        )

    client = RunwayML(api_key=api_key)

    # Build prompt: motion instruction first (most weight), then scene context
    if raw_prompt:
        # description is already a fully hand-crafted prompt — use as-is
        prompt = description
    else:
        base_motion = ACTION_TO_PROMPT.get(action, ACTION_TO_PROMPT["walk"])
        if description:
            short_desc = description[:100]
            prompt = f"{base_motion} Scene: {short_desc}. Keep the illustrated art style unchanged, no photorealism."
        else:
            prompt = f"{base_motion} Keep the illustrated art style unchanged, no photorealism."

    print(f"    Runway prompt: {prompt[:80]}...")

    # Upload image as data URI
    data_uri = _image_to_data_uri(image_path)

    task = client.image_to_video.create(
        model=RUNWAY_MODEL,
        prompt_image=data_uri,
        prompt_text=prompt,
        ratio="1280:720",
        duration=VIDEO_DURATION,
    )

    print(f"    Runway task submitted: {task.id} — waiting for render...")

    # Poll until complete
    while True:
        time.sleep(8)
        result = client.tasks.retrieve(task.id)
        status = result.status
        if status == "SUCCEEDED":
            break
        elif status in ("FAILED", "CANCELED"):
            raise RuntimeError(f"Runway task {task.id} ended with status: {status}")
        print(f"    Runway status: {status}...")

    # Download the MP4
    video_url = result.output[0]
    print(f"    Downloading Runway output...")
    urllib.request.urlretrieve(video_url, output_path)
    print(f"    Saved: {Path(output_path).name}")
    return output_path
