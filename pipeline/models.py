from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PagePlan:
    filename: str          # e.g. "page01.jpg"
    character: bool        # True if an animatable character was detected
    action: Optional[str]  # "walk", "wave", "jump", "balance", "sit", None
    description: Optional[str]  # human-readable description from vision AI


@dataclass
class BookPlan:
    title: str
    pages: List[PagePlan]
