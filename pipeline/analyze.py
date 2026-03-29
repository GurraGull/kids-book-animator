import json
import base64
import os
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
  "character": true or false,
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


def analyze_book(image_paths: list, title: str) -> BookPlan:
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
