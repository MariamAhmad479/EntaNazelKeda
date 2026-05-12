"""
wardrobe_manager.py
===================
Handles loading, saving, and querying the user's digital wardrobe.
The wardrobe is persisted as a JSON file on disk.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .data_models import ClothingCategory, ClothingItem, Occasion, Season, Style


class WardrobeManager:
    """Manages the collection of clothing items in the user's wardrobe.

    Parameters
    ----------
    wardrobe_path : str
        Path to the JSON file that stores wardrobe data.
    """

    def __init__(self, wardrobe_path: str):
        self.wardrobe_path = wardrobe_path
        self._items: Dict[str, ClothingItem] = {}
        if os.path.exists(wardrobe_path):
            self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load wardrobe from the JSON file on disk."""
        with open(self.wardrobe_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items_data = data if isinstance(data, list) else data.get("items", [])
        self._items = {}
        for item_dict in items_data:
            item = ClothingItem.from_dict(item_dict)
            self._items[item.id] = item

    def save(self) -> None:
        """Persist current wardrobe state to the JSON file."""
        os.makedirs(os.path.dirname(self.wardrobe_path) or ".", exist_ok=True)
        data = {"items": [item.to_dict() for item in self._items.values()]}
        with open(self.wardrobe_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def add_item(self, item: ClothingItem) -> str:
        """Add a clothing item to the wardrobe. Returns the item's id."""
        self._items[item.id] = item
        self.save()
        return item.id

    def add_item_from_dict(self, item_dict: dict) -> str:
        """Create a ClothingItem from a dict and add it. Returns item id."""
        item = ClothingItem.from_dict(item_dict)
        return self.add_item(item)

    def remove_item(self, item_id: str) -> bool:
        """Remove an item by id. Returns True if found and removed."""
        if item_id in self._items:
            del self._items[item_id]
            self.save()
            return True
        return False

    def get_item(self, item_id: str) -> Optional[ClothingItem]:
        """Retrieve a single item by id."""
        return self._items.get(item_id)

    def update_item(self, item_id: str, updates: dict) -> bool:
        """Update fields of an existing item. Returns True if successful."""
        item = self._items.get(item_id)
        if item is None:
            return False

        # Rebuild from merged dict
        current = item.to_dict()
        current.update(updates)
        current["id"] = item_id  # preserve original id
        self._items[item_id] = ClothingItem.from_dict(current)
        self.save()
        return True

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_all_items(self) -> List[ClothingItem]:
        """Return all items in the wardrobe."""
        return list(self._items.values())

    def get_items_by_category(self, category: ClothingCategory) -> List[ClothingItem]:
        """Return items matching a specific clothing category."""
        return [i for i in self._items.values() if i.category == category]

    def get_items_by_occasion(self, occasion: Occasion) -> List[ClothingItem]:
        """Return items suitable for a given occasion."""
        return [i for i in self._items.values() if occasion in i.occasions]

    def get_items_by_season(self, season: Season) -> List[ClothingItem]:
        """Return items suitable for a given season."""
        return [i for i in self._items.values() if season in i.seasons]

    def get_items_by_style(self, style: Style) -> List[ClothingItem]:
        """Return items matching a given style."""
        return [i for i in self._items.values() if i.style == style]

    def get_items_by_warmth(self, min_warmth: int, max_warmth: int) -> List[ClothingItem]:
        """Return items within a warmth-level range (inclusive)."""
        return [
            i for i in self._items.values()
            if min_warmth <= i.warmth_level <= max_warmth
        ]

    @property
    def size(self) -> int:
        """Number of items in the wardrobe."""
        return len(self._items)

    def summary(self) -> dict:
        """Quick summary of wardrobe composition by category."""
        counts: Dict[str, int] = {}
        for item in self._items.values():
            key = item.category.value
            counts[key] = counts.get(key, 0) + 1
        return {"total_items": self.size, "by_category": counts}

    def __repr__(self) -> str:
        return f"WardrobeManager(path={self.wardrobe_path!r}, items={self.size})"
