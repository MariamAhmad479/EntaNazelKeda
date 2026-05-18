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
    """Map temperature to a (min_warmth, max_warmth) range adjusted for Egypt's climate.

    | Temp (°C)  | Warmth range | Typical season   |
    |------------|-------------|------------------|
    | ≤ 12       | 4 – 5       | Winter           |
    | 13 – 20    | 3 – 4       | Autumn / Spring  |
    | 21 – 27    | 2 – 3       | Spring / Autumn  |
    | ≥ 28       | 1 – 2       | Summer           |
    """
    if temp_celsius <= 12:
        return (4, 5)
    elif temp_celsius <= 20:
        return (3, 4)
    elif temp_celsius <= 27:
        return (2, 3)
    else:
        return (1, 2)


def _temperature_to_seasons(temp_celsius: float) -> List[Season]:
    """Map temperature to likely seasons."""
    if temp_celsius <= 12:
        return [Season.WINTER]
    elif temp_celsius <= 20:
        return [Season.AUTUMN, Season.SPRING]
    elif temp_celsius <= 27:
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

    def _filter_slot(
        self,
        slot_items: List[ClothingItem],
        occasion: Optional[str] = None,
        weather: Optional[Dict] = None,
        style: Optional[str] = None,
        gender: Optional[str] = None,
        color: Optional[str] = None,
    ) -> List[ClothingItem]:
        """Apply filters to a specific slot with progressive relaxation."""
        if not slot_items:
            return []

        # 1. Apply Gender filter (strict, unless it leaves it empty)
        res = slot_items
        if gender is not None and gender.lower() != "unisex":
            res = self._filter_by_gender(res, gender)
            if not res:
                res = slot_items  # Fallback to unisex/original

        # 2. Apply Style filter (has its own fallback to all if no match)
        if style is not None:
            res = self._filter_by_style(res, style)

        # 3. Progressive relaxation levels for Color, Occasion, and Weather
        for col_level in ["strict", "any"]:
            for occ_level, wea_level in [
                ("strict", "strict"),
                ("strict", "relaxed"),
                ("strict", "any"),
                ("any", "strict"),
                ("any", "relaxed"),
                ("any", "any")
            ]:
                # Filter by occasion
                occ_filtered = res
                if occasion is not None and occ_level == "strict":
                    try:
                        occ_val = Occasion(occasion.lower())
                        occ_filtered = [item for item in res if occ_val in item.occasions]
                    except ValueError:
                        pass

                if not occ_filtered:
                    continue

                # Filter by weather
                wea_filtered = occ_filtered
                if weather is not None and weather.get("temperature") is not None:
                    temp = weather["temperature"]
                    min_w, max_w = _temperature_to_warmth_range(temp)
                    valid_seasons = set(_temperature_to_seasons(temp))

                    def _passes_wea(item: ClothingItem, level: str) -> bool:
                        # Season check (relaxed/strict both do season check)
                        item_seasons = set(item.seasons)
                        if item_seasons and not item_seasons.intersection(valid_seasons):
                            if len(item_seasons) < 4:
                                return False

                        # Warmth check
                        if level == "strict":
                            return min_w <= item.warmth_level <= max_w
                        elif level == "relaxed":
                            return min_w - 1 <= item.warmth_level <= max_w + 1
                        return True # "any"

                    wea_filtered = [item for item in occ_filtered if _passes_wea(item, wea_level)]

                    if not wea_filtered and wea_level == "any":
                        # Last resort: sort by closeness to target warmth center
                        target = (min_w + max_w) / 2
                        wea_filtered = sorted(occ_filtered, key=lambda i: abs(i.warmth_level - target))

                if not wea_filtered:
                    continue

                # Filter by color
                col_filtered = wea_filtered
                if color is not None and col_level == "strict":
                    col_filtered = [item for item in wea_filtered if color.lower() in item.color_name.lower()]

                if col_filtered:
                    return col_filtered

        return res

    def filter_items(
        self,
        items: List[ClothingItem],
        occasion: Optional[str] = None,
        weather: Optional[Dict] = None,
        style: Optional[str] = None,
        gender: Optional[str] = None,
        color: Optional[str] = None,
        piece: Optional[str] = None,
    ) -> List[ClothingItem]:
        """Apply all active filters and return matching items using slot-aware relaxation.

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
        color : str or None
            Preferred color name.
        piece : str or None
            Garment piece restriction name.
        """
        if not items:
            return []

        # Globally exclude any innerwear (socks, bra, underwear, briefs, sleepwear, loungewear, pajamas, etc.)
        innerwear_keywords = {
            "bra", "bras", "bralette", "bralettes", "underwear", "brief", "briefs",
            "panty", "panties", "trunk", "trunks", "boxer", "boxers", "socks", "sock",
            "tight", "tights", "stocking", "stockings", "nightwear", "sleepwear",
            "loungewear", "pajamas", "pyjamas", "undergarment", "undergarments"
        }
        
        import re
        result = [
            i for i in items
            if set(re.findall(r"[a-z]+", i.name.lower())).isdisjoint(innerwear_keywords)
        ]

        from .data_models import ClothingCategory, TOP_CATEGORIES, BOTTOM_CATEGORIES, FULL_BODY_CATEGORIES, FOOTWEAR_CATEGORIES

        # Map piece string to ClothingCategory
        piece_cat = None
        if piece is not None:
            p_lower = piece.lower()
            if "skirt" in p_lower:
                piece_cat = ClothingCategory.SKIRT
            elif "pant" in p_lower or "trouser" in p_lower or "jean" in p_lower:
                piece_cat = ClothingCategory.PANTS
            elif "short" in p_lower:
                piece_cat = ClothingCategory.SHORTS
            elif "dress" in p_lower or "gown" in p_lower:
                piece_cat = ClothingCategory.DRESS
            elif "shirt" in p_lower or "blouse" in p_lower or "top" in p_lower or "tshirt" in p_lower or "t-shirt" in p_lower:
                piece_cat = ClothingCategory.SHIRT
            elif "jacket" in p_lower or "coat" in p_lower or "blazer" in p_lower:
                piece_cat = ClothingCategory.JACKET
            elif "accessory" in p_lower or "bag" in p_lower or "hat" in p_lower or "sunglass" in p_lower:
                piece_cat = ClothingCategory.ACCESSORY
            elif "shoe" in p_lower or "sandal" in p_lower or "boot" in p_lower or "sneaker" in p_lower or "heel" in p_lower:
                piece_cat = ClothingCategory.SHOES

        # Partition into category slots
        shoes_items = [i for i in result if i.category in FOOTWEAR_CATEGORIES]
        dresses_items = [i for i in result if i.category in FULL_BODY_CATEGORIES]
        tops_items = [i for i in result if i.category in TOP_CATEGORIES and i.category != ClothingCategory.JACKET]
        bottoms_items = [i for i in result if i.category in BOTTOM_CATEGORIES]
        
        # If weather is hot (>= 25°C), do not suggest jackets at all to prevent sweat/heat issues
        is_hot_weather = False
        if weather is not None and weather.get("temperature") is not None:
            if weather["temperature"] >= 25:
                is_hot_weather = True
                
        jackets_items = [i for i in result if i.category == ClothingCategory.JACKET]
        if is_hot_weather:
            jackets_items = []  # Disallow jackets completely in hot summer weather!
        
        accessories_items = [i for i in result if i.category == ClothingCategory.ACCESSORY]
        
        # Restrict slots based on piece_cat
        if piece_cat is not None:
            if piece_cat in FOOTWEAR_CATEGORIES:
                shoes_items = [i for i in shoes_items if i.category == piece_cat]
            elif piece_cat in FULL_BODY_CATEGORIES:
                dresses_items = [i for i in dresses_items if i.category == piece_cat]
                tops_items = []
                bottoms_items = []
            elif piece_cat == ClothingCategory.SHIRT:
                tops_items = [i for i in tops_items if i.category == piece_cat]
                dresses_items = []
            elif piece_cat in BOTTOM_CATEGORIES:
                bottoms_items = [i for i in bottoms_items if i.category == piece_cat]
                dresses_items = []
            elif piece_cat == ClothingCategory.JACKET:
                jackets_items = [i for i in jackets_items if i.category == piece_cat]
            elif piece_cat == ClothingCategory.ACCESSORY:
                accessories_items = [i for i in accessories_items if i.category == piece_cat]

        captured_ids = {i.id for i in shoes_items + dresses_items + tops_items + bottoms_items + jackets_items + accessories_items}
        other_items = [i for i in result if i.id not in captured_ids]

        # Apply progressive filtering to each slot individually
        filtered_shoes = self._filter_slot(shoes_items, occasion, weather, style, gender, color)
        filtered_dresses = self._filter_slot(dresses_items, occasion, weather, style, gender, color)
        filtered_tops = self._filter_slot(tops_items, occasion, weather, style, gender, color)
        filtered_bottoms = self._filter_slot(bottoms_items, occasion, weather, style, gender, color)
        
        filtered_jackets = self._filter_slot(jackets_items, occasion, weather, style, gender, color)
        filtered_accessories = self._filter_slot(accessories_items, occasion, weather, style, gender, color)
        filtered_other = self._filter_slot(other_items, occasion, weather, style, gender, color)

        return filtered_shoes + filtered_dresses + filtered_tops + filtered_bottoms + filtered_jackets + filtered_accessories + filtered_other

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
