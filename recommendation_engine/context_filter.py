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
        gender: Optional[str] = None,
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
        gender : str or None
            Gender for styling ("Male", "Female", "Unisex").
        """
        result = list(items)

        # Globally exclude any innerwear (socks, bra, underwear, briefs, sleepwear, loungewear, pajamas, etc.)
        innerwear_keywords = {
            "bra", "bras", "bralette", "bralettes", "underwear", "brief", "briefs",
            "panty", "panties", "trunk", "trunks", "boxer", "boxers", "socks", "sock",
            "tight", "tights", "stocking", "stockings", "nightwear", "sleepwear",
            "loungewear", "pajamas", "pyjamas", "undergarment", "undergarments"
        }
        
        import re
        result = [
            i for i in result
            if set(re.findall(r"[a-z]+", i.name.lower())).isdisjoint(innerwear_keywords)
        ]

        if occasion is not None:
            result = self._filter_by_occasion(result, occasion)

        if weather is not None:
            result = self._filter_by_weather(result, weather)

        if style is not None:
            result = self._filter_by_style(result, style)

        if gender is not None and gender.lower() != "unisex":
            result = self._filter_by_gender(result, gender)

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

        Strategy
        --------
        1. Strict pass  — items within the exact warmth range + matching season.
        2. Relaxed pass — ±1 warmth tolerance, used only if strict yields < 3 items.
        3. Last resort  — pick items closest to the target warmth (never silently
           returns all items, which was causing shorts to appear in cold weather).
        """
        temp = weather.get("temperature")

        if temp is None:
            return items

        min_warmth, max_warmth = _temperature_to_warmth_range(temp)
        valid_seasons = set(_temperature_to_seasons(temp))

        def _passes(item: ClothingItem, strict: bool) -> bool:
            # ── Warmth check ──────────────────────────────────────────
            in_range = min_warmth <= item.warmth_level <= max_warmth
            if not in_range:
                if strict:
                    return False
                # Relaxed: allow ±1
                if not (min_warmth - 1 <= item.warmth_level <= max_warmth + 1):
                    return False

            # ── Season check ─────────────────────────────────────────
            item_seasons = set(item.seasons)
            if item_seasons and not item_seasons.intersection(valid_seasons):
                # Year-round items (4 seasons listed) always pass
                if len(item_seasons) < 4:
                    return False

            return True

        # Pass 1: strict
        filtered = [item for item in items if _passes(item, strict=True)]

        # Pass 2: relax warmth ±1 if strict left too few
        if len(filtered) < 3:
            relaxed = [item for item in items if _passes(item, strict=False)]
            if len(relaxed) > len(filtered):
                filtered = relaxed

        # Pass 3: last resort — rank by closeness to target warmth centre
        if not filtered:
            target = (min_warmth + max_warmth) / 2
            sorted_by_closeness = sorted(items, key=lambda i: abs(i.warmth_level - target))
            filtered = sorted_by_closeness[:max(3, len(items) // 2)]

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

    # ------------------------------------------------------------------
    # Gender filter
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_by_gender(
        items: List[ClothingItem], gender: str
    ) -> List[ClothingItem]:
        gender_lower = gender.lower()
        if gender_lower not in ("female", "male"):
            return items

        filtered = []
        female_exclude = ["men", "boy", "tie", "cufflink", "trunk", "boxer", "brief"]
        male_exclude = ["women", "girl", "dress", "skirt", "saree", "blouse", "legging", "heel", "purse", "bra", "panties"]

        for i in items:
            item_gender = getattr(i, "gender", "unisex").lower()
            
            # Check explicit metadata gender first
            if item_gender == "female":
                if gender_lower == "female":
                    filtered.append(i)
            elif item_gender == "male":
                if gender_lower == "male":
                    filtered.append(i)
            else:
                # Unisex / Custom User Uploads fallback: apply substring name rules
                name_lower = i.name.lower()
                if gender_lower == "female":
                    if not any(w in name_lower for w in female_exclude):
                        filtered.append(i)
                elif gender_lower == "male":
                    if not any(w in name_lower for w in male_exclude):
                        filtered.append(i)
                        
        return filtered

    def __repr__(self) -> str:
        return "ContextFilter()"
