"""
vision_bridge.py
================
Converts Person 1's vision model artifacts (clothing_metadata.csv +
clothing_features.npy) into ClothingItem objects that the recommendation
engine can consume directly.

This is the integration layer between the vision module and the
recommendation engine — it maps Kaggle Fashion Dataset fields to the
ClothingItem schema without modifying either side.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_models import (
    ClothingCategory,
    ClothingItem,
    Occasion,
    Pattern,
    Season,
    Style,
)


# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

# articleType → ClothingCategory
ARTICLE_TYPE_MAP: Dict[str, ClothingCategory] = {
    # Tops
    "Tshirts": ClothingCategory.SHIRT,
    "Shirts": ClothingCategory.SHIRT,
    "Tops": ClothingCategory.SHIRT,
    # Bottoms
    "Jeans": ClothingCategory.PANTS,
    "Trousers": ClothingCategory.PANTS,
    "Track Pants": ClothingCategory.PANTS,
    "Leggings": ClothingCategory.PANTS,
    # Shorts
    "Shorts": ClothingCategory.SHORTS,
    # Skirts
    "Skirts": ClothingCategory.SKIRT,
    # Dresses
    "Dresses": ClothingCategory.DRESS,
    "Jumpsuit": ClothingCategory.DRESS,
    # Footwear
    "Casual Shoes": ClothingCategory.SHOES,
    "Sports Shoes": ClothingCategory.SHOES,
    "Heels": ClothingCategory.SHOES,
    "Sandals": ClothingCategory.SHOES,
    "Flip Flops": ClothingCategory.SHOES,
    "Flats": ClothingCategory.SHOES,
    # Outerwear → jacket
    "Blazers": ClothingCategory.JACKET,
    "Jackets": ClothingCategory.JACKET,
    "Sweatshirts": ClothingCategory.JACKET,
    "Sweaters": ClothingCategory.JACKET,
    # Everything else → accessory
}

# baseColour → (R, G, B)
COLOUR_RGB_MAP: Dict[str, Tuple[int, int, int]] = {
    "black": (20, 20, 20),
    "white": (245, 245, 245),
    "off white": (255, 250, 240),
    "grey": (160, 160, 160),
    "grey melange": (180, 180, 180),
    "charcoal": (54, 69, 79),
    "navy blue": (0, 0, 128),
    "blue": (70, 130, 180),
    "dark blue": (25, 25, 112),
    "light blue": (173, 216, 230),
    "turquoise blue": (0, 206, 209),
    "teal": (0, 128, 128),
    "red": (180, 30, 30),
    "maroon": (128, 0, 0),
    "burgundy": (128, 0, 32),
    "pink": (255, 182, 193),
    "magenta": (255, 0, 144),
    "lavender": (180, 160, 220),
    "purple": (128, 0, 128),
    "green": (34, 139, 34),
    "olive": (107, 142, 35),
    "khaki": (195, 176, 145),
    "yellow": (255, 215, 0),
    "orange": (255, 140, 0),
    "brown": (139, 90, 43),
    "coffee brown": (75, 54, 33),
    "tan": (210, 180, 140),
    "beige": (210, 180, 140),
    "cream": (255, 253, 208),
    "gold": (212, 175, 55),
    "silver": (192, 192, 192),
    "copper": (184, 115, 51),
    "bronze": (205, 127, 50),
    "steel": (113, 121, 126),
    "nude": (227, 187, 164),
    "peach": (255, 218, 185),
    "coral": (255, 127, 80),
    "rust": (183, 65, 14),
    "lime green": (50, 205, 50),
    "sea green": (46, 139, 87),
    "fluorescent green": (0, 255, 0),
    "multi": (128, 128, 128),        # neutral fallback for multi-colour
    "mushroom brown": (150, 120, 90),
    "taupe": (72, 60, 50),
    "mauve": (224, 176, 255),
    "skin": (227, 187, 164),
}

# usage → list of Occasion values
USAGE_OCCASION_MAP: Dict[str, List[Occasion]] = {
    "casual": [Occasion.CASUAL],
    "formal": [Occasion.FORMAL, Occasion.BUSINESS],
    "sports": [Occasion.SPORT, Occasion.OUTDOOR],
    "ethnic": [Occasion.CASUAL, Occasion.PARTY],
    "party": [Occasion.PARTY, Occasion.CASUAL],
    "smart casual": [Occasion.CASUAL, Occasion.BUSINESS],
    "travel": [Occasion.CASUAL, Occasion.OUTDOOR],
    "home": [Occasion.CASUAL],
}

# season → Season enum
SEASON_MAP: Dict[str, Season] = {
    "summer": Season.SUMMER,
    "winter": Season.WINTER,
    "fall": Season.AUTUMN,
    "spring": Season.SPRING,
}

# Keywords in productDisplayName for pattern inference
PATTERN_KEYWORDS: Dict[str, Pattern] = {
    "check": Pattern.PLAID,
    "checked": Pattern.PLAID,
    "plaid": Pattern.PLAID,
    "stripe": Pattern.STRIPED,
    "striped": Pattern.STRIPED,
    "stripes": Pattern.STRIPED,
    "printed": Pattern.GRAPHIC,
    "print": Pattern.GRAPHIC,
    "graphic": Pattern.GRAPHIC,
    "floral": Pattern.FLORAL,
    "polka": Pattern.GRAPHIC,
    "abstract": Pattern.ABSTRACT,
    "camouflage": Pattern.ABSTRACT,
}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer_pattern(product_name: str) -> Pattern:
    """Infer pattern from the product display name."""
    name_lower = product_name.lower()
    for keyword, pattern in PATTERN_KEYWORDS.items():
        if keyword in name_lower:
            return pattern
    return Pattern.SOLID


def _infer_style(usage: str, article_type: str, category: ClothingCategory) -> Style:
    """Infer style from usage and article type."""
    usage_lower = usage.lower()
    article_lower = article_type.lower()

    if usage_lower == "sports":
        return Style.ATHLETIC
    if usage_lower == "ethnic":
        return Style.BOHEMIAN
    if usage_lower == "formal":
        return Style.CLASSIC
    if usage_lower == "party":
        return Style.CLASSIC

    # Casual inference based on article type
    if article_lower in ("tshirts", "shorts", "flip flops"):
        return Style.STREETWEAR
    if article_lower in ("blazers", "shirts", "trousers", "heels"):
        return Style.CLASSIC
    if category == ClothingCategory.SHOES:
        if "sports" in article_lower or "casual" in article_lower:
            return Style.STREETWEAR
        return Style.CLASSIC

    return Style.CLASSIC


def _infer_warmth(season: str, article_type: str, category: ClothingCategory) -> int:
    """Infer warmth level (1-5) from season and article type."""
    article_lower = article_type.lower()
    season_lower = season.lower() if season else ""

    # Heavy outerwear
    if article_lower in ("jackets", "sweaters", "sweatshirts", "blazers"):
        if season_lower == "winter":
            return 4
        return 3

    # Light items
    if article_lower in ("tshirts", "tops", "flip flops", "sandals", "shorts"):
        return 1

    # Seasonal adjustment
    if season_lower == "winter":
        return 3
    if season_lower in ("fall", "spring"):
        return 2
    return 1  # summer default


def _infer_formality(usage: str, article_type: str) -> int:
    """Infer formality level (1-5) from usage and article type."""
    usage_lower = usage.lower()
    article_lower = article_type.lower()

    if usage_lower == "formal":
        if article_lower in ("shirts", "trousers", "blazers", "heels"):
            return 5
        return 4
    if usage_lower == "party":
        if article_lower in ("dresses", "heels"):
            return 4
        return 3
    if usage_lower == "sports":
        return 1
    if usage_lower == "ethnic":
        return 3

    # Casual
    if article_lower in ("tshirts", "shorts", "flip flops", "sandals"):
        return 1
    if article_lower in ("shirts", "trousers"):
        return 2
    if article_lower in ("jeans", "casual shoes"):
        return 2
    return 2


def _get_colour_rgb(colour_name: str) -> Tuple[int, int, int]:
    """Look up RGB for a colour name, with fallback to neutral grey."""
    return COLOUR_RGB_MAP.get(colour_name.lower().strip(), (128, 128, 128))


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def load_vision_items(
    vision_dir: str,
    metadata_file: str = "clothing_metadata.csv",
    features_file: str = "clothing_features.npy",
) -> List[ClothingItem]:
    """Load Person 1's vision artifacts and convert to ClothingItem objects.

    Parameters
    ----------
    vision_dir : str
        Path to the ``vision/saved/`` directory containing the CSV and NPY.
    metadata_file : str
        Name of the metadata CSV file.
    features_file : str
        Name of the NumPy features file.

    Returns
    -------
    list[ClothingItem]
        Fully populated clothing items ready for the recommendation engine.
    """
    csv_path = os.path.join(vision_dir, metadata_file)
    npy_path = os.path.join(vision_dir, features_file)

    # Load the feature vectors (shape: N × 2048)
    features: Optional[np.ndarray] = None
    if os.path.exists(npy_path):
        features = np.load(npy_path)

    # Read metadata CSV
    items: List[ClothingItem] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            item = _convert_row(row, row_idx, features)
            if item is not None:
                items.append(item)

    return items


def _convert_row(
    row: Dict[str, str],
    row_idx: int,
    features: Optional[np.ndarray],
) -> Optional[ClothingItem]:
    """Convert a single CSV row to a ClothingItem.

    Returns None if the row has missing essential data.
    """
    article_type = row.get("articleType", "").strip()
    base_colour = row.get("baseColour", "").strip()
    usage = row.get("usage", "Casual").strip()
    season = row.get("season", "").strip()
    product_name = row.get("productDisplayName", "").strip()
    image_path = row.get("image_path", "").strip()
    item_id = row.get("id", "").strip()

    if not article_type or not product_name:
        return None

    # Category
    category = ARTICLE_TYPE_MAP.get(article_type, ClothingCategory.ACCESSORY)

    # Colour
    colour_name = base_colour.lower() if base_colour else "grey"
    colour_rgb = _get_colour_rgb(colour_name)

    # Occasions
    occasions = USAGE_OCCASION_MAP.get(usage.lower(), [Occasion.CASUAL])

    # Seasons
    if season and season.lower() in SEASON_MAP:
        seasons = [SEASON_MAP[season.lower()]]
    else:
        # Unknown season → all-season
        seasons = [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]

    # Inferred fields
    pattern = _infer_pattern(product_name)
    style = _infer_style(usage, article_type, category)
    warmth = _infer_warmth(season, article_type, category)
    formality = _infer_formality(usage, article_type)

    # Image features from the NPY array
    image_feats: Optional[List[float]] = None
    if features is not None and row_idx < len(features):
        image_feats = features[row_idx].tolist()

    return ClothingItem(
        id=f"v{item_id}" if item_id else f"v{row_idx:05d}",
        name=product_name,
        category=category,
        color_rgb=colour_rgb,
        color_name=colour_name,
        pattern=pattern,
        style=style,
        occasions=occasions,
        seasons=seasons,
        warmth_level=warmth,
        formality_level=formality,
        image_features=image_feats,
        image_path=image_path,
    )
