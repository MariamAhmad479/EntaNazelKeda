"""
data_models.py
==============
Core data structures for the Smart Wardrobe recommendation engine.

Defines enums for clothing categories, occasions, seasons, styles, and
patterns, plus the ClothingItem dataclass that represents a single
garment in the user's wardrobe.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ClothingCategory(Enum):
    """High-level garment type."""
    SHIRT = "shirt"
    PANTS = "pants"
    DRESS = "dress"
    SHOES = "shoes"
    JACKET = "jacket"
    SKIRT = "skirt"
    ACCESSORY = "accessory"
    SHORTS = "shorts"


class Occasion(Enum):
    """When / where the item is appropriate."""
    CASUAL = "casual"
    FORMAL = "formal"
    BUSINESS = "business"
    SPORT = "sport"
    PARTY = "party"
    OUTDOOR = "outdoor"


class Season(Enum):
    """Seasons the item is suited for."""
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


class Style(Enum):
    """Aesthetic style tags."""
    CLASSIC = "classic"
    STREETWEAR = "streetwear"
    BOHEMIAN = "bohemian"
    MINIMALIST = "minimalist"
    PREPPY = "preppy"
    ATHLETIC = "athletic"


class Pattern(Enum):
    """Visual pattern / print of the fabric."""
    SOLID = "solid"
    STRIPED = "striped"
    PLAID = "plaid"
    FLORAL = "floral"
    GRAPHIC = "graphic"
    ABSTRACT = "abstract"


# ---------------------------------------------------------------------------
# Slot helpers  (used by outfit generator)
# ---------------------------------------------------------------------------

# Categories that fill the "top" slot
TOP_CATEGORIES = {ClothingCategory.SHIRT, ClothingCategory.JACKET}

# Categories that fill the "bottom" slot
BOTTOM_CATEGORIES = {ClothingCategory.PANTS, ClothingCategory.SKIRT, ClothingCategory.SHORTS}

# Categories that act as a full-body item (replaces top + bottom)
FULL_BODY_CATEGORIES = {ClothingCategory.DRESS}

# Categories for the feet slot
FOOTWEAR_CATEGORIES = {ClothingCategory.SHOES}

# Optional add-ons
OPTIONAL_CATEGORIES = {ClothingCategory.ACCESSORY, ClothingCategory.JACKET}


# ---------------------------------------------------------------------------
# Main data class
# ---------------------------------------------------------------------------

@dataclass
class ClothingItem:
    """Represents a single garment in the user's wardrobe.

    Attributes
    ----------
    id : str
        Unique identifier (auto-generated UUID if not supplied).
    name : str
        Human-readable name, e.g. "Navy Oxford Shirt".
    category : ClothingCategory
        What kind of garment this is.
    color_rgb : tuple[int, int, int]
        Dominant colour as (R, G, B) in 0-255.
    color_name : str
        Human-readable colour label, e.g. "navy blue".
    pattern : Pattern
        Fabric pattern / print.
    style : Style
        Aesthetic style tag.
    occasions : list[Occasion]
        When / where the item is appropriate.
    seasons : list[Season]
        Suitable seasons.
    warmth_level : int
        1 (very light) to 5 (very warm).
    formality_level : int
        1 (very casual) to 5 (very formal).
    image_features : list[float] | None
        Optional CNN embedding vector from Person 1's image model.
    image_path : str | None
        Optional path to the garment image.
    """

    name: str
    category: ClothingCategory
    color_rgb: tuple
    color_name: str
    pattern: Pattern
    style: Style
    occasions: List[Occasion]
    seasons: List[Season]
    warmth_level: int  # 1-5
    formality_level: int  # 1-5
    gender: str = "unisex"
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    image_features: Optional[List[float]] = None
    image_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "color_rgb": list(self.color_rgb),
            "color_name": self.color_name,
            "pattern": self.pattern.value,
            "style": self.style.value,
            "occasions": [o.value for o in self.occasions],
            "seasons": [s.value for s in self.seasons],
            "warmth_level": self.warmth_level,
            "formality_level": self.formality_level,
            "gender": self.gender,
            "image_features": self.image_features,
            "image_path": self.image_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClothingItem":
        """Construct a ClothingItem from a dictionary (e.g. loaded from JSON)."""
        return cls(
            id=d.get("id", str(uuid.uuid4())[:8]),
            name=d["name"],
            category=ClothingCategory(d["category"]),
            color_rgb=tuple(d["color_rgb"]),
            color_name=d["color_name"],
            pattern=Pattern(d["pattern"]),
            style=Style(d["style"]),
            occasions=[Occasion(o) for o in d["occasions"]],
            seasons=[Season(s) for s in d["seasons"]],
            warmth_level=d["warmth_level"],
            formality_level=d["formality_level"],
            gender=d.get("gender", "unisex"),
            image_features=d.get("image_features"),
            image_path=d.get("image_path"),
        )

    def __repr__(self) -> str:
        return (
            f"ClothingItem(id={self.id!r}, name={self.name!r}, "
            f"category={self.category.value}, color={self.color_name})"
        )
