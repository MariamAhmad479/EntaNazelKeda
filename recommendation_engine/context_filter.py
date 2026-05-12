"""
context_filter.py
=================
Filters wardrobe items and outfits based on contextual constraints:
    - Occasion  (casual, formal, business, sport, party, outdoor)
    - Weather   (temperature → warmth level + season mapping)
    - Style     (classic, streetwear, minimalist, etc.)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .data_models import ClothingItem, Occasion, Season, Style


# ---------------------------------------------------------------------------
# Temperature → warmth / season mapping
# ---------------------------------------------------------------------------

def _temperature_to_warmth_range(temp_celsius: float) -> tuple:
    """Map temperature to a (min_warmth, max_warmth) range.

    | Temp (°C)  | Warmth range | Typical season   |
    |------------|-------------|------------------|
    | ≤ 5        | 4 – 5       | Winter           |
    | 6 – 15     | 3 – 4       | Autumn / Spring  |
    | 16 – 24    | 2 – 3       | Spring / Autumn  |
    | ≥ 25       | 1 – 2       | Summer           |
    """
    if temp_celsius <= 5:
        return (4, 5)
    elif temp_celsius <= 15:
        return (3, 4)
    elif temp_celsius <= 24:
        return (2, 3)
    else:
        return (1, 2)


def _temperature_to_seasons(temp_celsius: float) -> List[Season]:
    """Map temperature to likely seasons."""
    if temp_celsius <= 5:
        return [Season.WINTER]
    elif temp_celsius <= 15:
        return [Season.AUTUMN, Season.SPRING]
    elif temp_celsius <= 24:
        return [Season.SPRING, Season.AUTUMN]
    else:
        return [Season.SUMMER]


# ---------------------------------------------------------------------------
# Main filter class
# ---------------------------------------------------------------------------

class ContextFilter:
    """Filter wardrobe items based on occasion, weather, and style.

    Usage
    -----
    >>> cf = ContextFilter()
    >>> filtered = cf.filter_items(
    ...     items,
    ...     occasion="formal",
    ...     weather={"temperature": 22, "condition": "sunny"},
    ...     style="classic",
    ... )
    """

    def filter_items(
        self,
        items: List[ClothingItem],
        occasion: Optional[str] = None,
        weather: Optional[Dict] = None,
        style: Optional[str] = None,
    ) -> List[ClothingItem]:
        """Apply all active filters and return matching items.

        Parameters
        ----------
        items : list[ClothingItem]
            Pool of items to filter.
        occasion : str or None
            Occasion name (must match an ``Occasion`` enum value).
        weather : dict or None
            ``{"temperature": float, "condition": str}``
            where condition is e.g. "sunny", "rainy", "snowy".
        style : str or None
            Style name (must match a ``Style`` enum value).
        """
        result = list(items)

        if occasion is not None:
            result = self._filter_by_occasion(result, occasion)

        if weather is not None:
            result = self._filter_by_weather(result, weather)

        if style is not None:
            result = self._filter_by_style(result, style)

        return result

    # ------------------------------------------------------------------
    # Occasion filter
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_by_occasion(
        items: List[ClothingItem], occasion: str
    ) -> List[ClothingItem]:
        """Keep items whose occasions list includes the requested one."""
        try:
            occ = Occasion(occasion.lower())
        except ValueError:
            # Unknown occasion — return all items rather than crash
            return items

        return [item for item in items if occ in item.occasions]

    # ------------------------------------------------------------------
    # Weather filter
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_by_weather(
        items: List[ClothingItem], weather: Dict
    ) -> List[ClothingItem]:
        """Filter items by temperature and weather condition.

        Uses warmth-level and season mapping.
        """
        temp = weather.get("temperature")
        condition = weather.get("condition", "").lower()

        if temp is None:
            return items

        min_warmth, max_warmth = _temperature_to_warmth_range(temp)
        valid_seasons = set(_temperature_to_seasons(temp))

        filtered = []
        for item in items:
            # Check warmth range
            if not (min_warmth <= item.warmth_level <= max_warmth):
                # Allow ±1 tolerance for flexibility
                if abs(item.warmth_level - min_warmth) > 1 and abs(item.warmth_level - max_warmth) > 1:
                    continue

            # Check season overlap
            item_seasons = set(item.seasons)
            if item_seasons and not item_seasons.intersection(valid_seasons):
                # Allow items that are multi-season (≥3 seasons = year-round)
                if len(item_seasons) < 3:
                    continue

            filtered.append(item)

        # If filtering removed everything, relax and return originals
        if not filtered:
            return items

        return filtered

    # ------------------------------------------------------------------
    # Style filter
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_by_style(
        items: List[ClothingItem], style: str
    ) -> List[ClothingItem]:
        """Keep items matching the requested style.

        Falls back to all items if no matches are found.
        """
        try:
            target = Style(style.lower())
        except ValueError:
            return items

        matches = [item for item in items if item.style == target]
        return matches if matches else items

    def __repr__(self) -> str:
        return "ContextFilter()"
