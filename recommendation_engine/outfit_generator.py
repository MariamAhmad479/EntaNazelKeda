"""
outfit_generator.py
===================
Generates candidate outfits from the wardrobe by combining items
across required clothing slots, then ranks them by compatibility.

Outfit slot rules
-----------------
- Standard outfit: **top** + **bottom** + **shoes**
- Dress outfit:    **dress** + **shoes**
- Optional extras: jacket, accessory (added if they improve the score)
"""

from __future__ import annotations

import uuid
from itertools import product
from typing import Dict, List, Optional, Tuple

from .compatibility import CompatibilityScorer
from .data_models import (
    BOTTOM_CATEGORIES,
    FULL_BODY_CATEGORIES,
    FOOTWEAR_CATEGORIES,
    OPTIONAL_CATEGORIES,
    TOP_CATEGORIES,
    ClothingCategory,
    ClothingItem,
)


class Outfit:
    """A scored combination of clothing items."""

    def __init__(self, items: List[ClothingItem], score: float = 0.0):
        self.id = str(uuid.uuid4())[:8]
        self.items = items
        self.score = score
        self._breakdown: Dict[str, float] = {}

    @property
    def breakdown(self) -> Dict[str, float]:
        return self._breakdown

    @breakdown.setter
    def breakdown(self, value: Dict[str, float]):
        self._breakdown = value

    def to_dict(self) -> dict:
        return {
            "outfit_id": self.id,
            "score": round(self.score, 4),
            "breakdown": {k: round(v, 4) for k, v in self._breakdown.items()},
            "items": [item.to_dict() for item in self.items],
            "summary": " + ".join(f"{i.name} ({i.color_name})" for i in self.items),
        }

    def __repr__(self) -> str:
        names = ", ".join(i.name for i in self.items)
        return f"Outfit(score={self.score:.3f}, items=[{names}])"


class OutfitGenerator:
    """Generate and rank candidate outfits from a wardrobe.

    Parameters
    ----------
    scorer : CompatibilityScorer or None
        If None, a default scorer is created.
    max_combinations : int
        Hard cap on how many candidate outfits to evaluate
        (prevents combinatorial explosion on large wardrobes).
    min_score : float
        Minimum compatibility score to keep an outfit.
    """

    def __init__(
        self,
        scorer: Optional[CompatibilityScorer] = None,
        max_combinations: int = 5000,
        min_score: float = 0.3,
    ):
        self.scorer = scorer or CompatibilityScorer()
        self.max_combinations = max_combinations
        self.min_score = min_score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        items: List[ClothingItem],
        top_n: int = 10,
        include_optional: bool = True,
        min_score: Optional[float] = None,
    ) -> List[Outfit]:
        """Generate the top-N outfits from the given items.

        Returns outfits sorted by descending compatibility score.
        """
        # Determine active min score threshold
        active_min_score = min_score if min_score is not None else self.min_score

        # Partition items by slot (shuffle a copy of the list first to guarantee variety)
        import random
        shuffled_items = list(items)
        random.shuffle(shuffled_items)

        tops = [i for i in shuffled_items if i.category in TOP_CATEGORIES and i.category != ClothingCategory.JACKET]
        bottoms = [i for i in shuffled_items if i.category in BOTTOM_CATEGORIES]
        dresses = [i for i in shuffled_items if i.category in FULL_BODY_CATEGORIES]
        shoes = [i for i in shuffled_items if i.category in FOOTWEAR_CATEGORIES]
        jackets = [i for i in shuffled_items if i.category == ClothingCategory.JACKET]
        accessories = [i for i in shuffled_items if i.category == ClothingCategory.ACCESSORY]

        candidates: List[Outfit] = []

        # --- Standard outfits: top + bottom + shoes ---
        if tops and bottoms and shoes:
            # Prevent combinatorial explosion BEFORE materializing the list
            num_combos = len(tops) * len(bottoms) * len(shoes)
            
            # If num_combos is huge, we don't want to materialize the whole list
            # We'll either sample directly or truncate the source lists
            if num_combos > self.max_combinations * 2:
                # Use a more efficient sampling approach or just truncate
                if num_combos > 100_000:
                    # Truncate to keep the search space reasonable
                    limit = 40 # 40^3 = 64,000
                    tops = tops[:limit]
                    bottoms = bottoms[:limit]
                    shoes = shoes[:limit]
                    num_combos = len(tops) * len(bottoms) * len(shoes)

                # Still sample to stay within max_combinations
                import random
                
                # To avoid materializing the list, we sample indices without global seeding
                indices = random.sample(range(num_combos), min(num_combos, self.max_combinations))
                for idx in indices:
                    # Map flat index back to product indices
                    i_shoe = idx % len(shoes)
                    idx //= len(shoes)
                    i_bottom = idx % len(bottoms)
                    idx //= len(bottoms)
                    i_top = idx % len(tops)
                    
                    base_items = [tops[i_top], bottoms[i_bottom], shoes[i_shoe]]
                    outfit = self._score_outfit(base_items)
                    if outfit.score >= active_min_score:
                        candidates.append(outfit)
            else:
                # Small enough to materialize
                combos = list(product(tops, bottoms, shoes))
                for top, bottom, shoe in combos:
                    base_items = [top, bottom, shoe]
                    outfit = self._score_outfit(base_items)
                    if outfit.score >= active_min_score:
                        candidates.append(outfit)

        # --- Dress outfits: dress + shoes ---
        if dresses and shoes:
            num_dress_combos = len(dresses) * len(shoes)
            
            # Prevent combinatorial explosion for dress combinations
            if num_dress_combos > self.max_combinations * 2:
                if num_dress_combos > 100_000:
                    # Truncate source lists to keep the search space reasonable
                    limit = 200  # 200^2 = 40,000
                    dresses = dresses[:limit]
                    shoes = shoes[:limit]
                    num_dress_combos = len(dresses) * len(shoes)
                
                import random
                indices = random.sample(range(num_dress_combos), min(num_dress_combos, self.max_combinations))
                for idx in indices:
                    i_shoe = idx % len(shoes)
                    i_dress = idx // len(shoes)
                    
                    base_items = [dresses[i_dress], shoes[i_shoe]]
                    outfit = self._score_outfit(base_items)
                    if outfit.score >= active_min_score:
                        candidates.append(outfit)
            else:
                for dress, shoe in product(dresses, shoes):
                    base_items = [dress, shoe]
                    outfit = self._score_outfit(base_items)
                    if outfit.score >= active_min_score:
                        candidates.append(outfit)

        # Sort by score descending
        candidates.sort(key=lambda o: o.score, reverse=True)

        # Optionally try adding jackets / accessories to top outfits
        if include_optional and candidates:
            # Only consider the top few jackets/accessories to save time
            candidates = self._try_add_optionals(
                candidates[:max(200, top_n * 10)], jackets[:30], accessories[:30]
            )
            candidates.sort(key=lambda o: o.score, reverse=True)

        # Explicitly ensure no outfit contains both a dress and a bottom, and deduplicate identical-looking outfits
        valid_candidates = []
        seen_outfits = set()
        for outfit in candidates:
            cats = {item.category for item in outfit.items}
            if ClothingCategory.DRESS in cats and (cats & BOTTOM_CATEGORIES):
                continue
            
            # Deduplicate by item names/colors so they look unique to the user
            outfit_key = tuple(sorted((item.name, item.color_name) for item in outfit.items))
            if outfit_key in seen_outfits:
                continue
            seen_outfits.add(outfit_key)
            valid_candidates.append(outfit)

        # Helper to check if an outfit has more than 2 items repeated with any already selected outfit
        def _has_too_much_overlap(outfit: Outfit, selected_outfits: List[Outfit]) -> bool:
            outfit_item_ids = {item.id for item in outfit.items}
            for selected in selected_outfits:
                selected_item_ids = {item.id for item in selected.items}
                if len(outfit_item_ids.intersection(selected_item_ids)) > 2:
                    return True
            return False

        # ── Smart Multi-Pass Greedy Diversity Reranker ──
        final_selection = []
        used_main_items = set()  # Track unique dresses, shirts, jackets, pants, skirts, shorts
        selected_bottom_cats = {}  # Track counts of BOTTOM categories (pants, skirt, shorts)
        selected_dresses_count = 0

        # Pass 1: Strict unique items, strict at most 1 dress, alternating standard bottoms (first time seeing each bottom cat)
        for outfit in valid_candidates:
            if len(final_selection) >= top_n:
                break
                
            is_dress = any(item.category == ClothingCategory.DRESS for item in outfit.items)
            
            # Dress constraint: at most 1 dress outfit in final recommendations
            if is_dress and selected_dresses_count >= 1:
                continue

            if _has_too_much_overlap(outfit, final_selection):
                continue

            main_items = [
                item.id for item in outfit.items 
                if item.category in (ClothingCategory.DRESS, ClothingCategory.SHIRT, ClothingCategory.JACKET, ClothingCategory.PANTS, ClothingCategory.SKIRT, ClothingCategory.SHORTS)
            ]
            has_overlap = any(item_id in used_main_items for item_id in main_items)
            if has_overlap:
                continue

            if not is_dress:
                bottom_item = next((item for item in outfit.items if item.category in BOTTOM_CATEGORIES), None)
                bottom_cat = bottom_item.category if bottom_item else None
                # In Pass 1, only allow a bottom category if it has been selected 0 times
                if bottom_cat and selected_bottom_cats.get(bottom_cat, 0) > 0:
                    continue

            # If we reach here, we accept it
            final_selection.append(outfit)
            if is_dress:
                selected_dresses_count += 1
            else:
                bottom_item = next((item for item in outfit.items if item.category in BOTTOM_CATEGORIES), None)
                bottom_cat = bottom_item.category if bottom_item else None
                if bottom_cat:
                    selected_bottom_cats[bottom_cat] = selected_bottom_cats.get(bottom_cat, 0) + 1
            for item_id in main_items:
                used_main_items.add(item_id)

        # Pass 2: Strict unique items, strict at most 1 dress, relax bottom category restriction (allow minimum counts)
        if len(final_selection) < top_n:
            for outfit in valid_candidates:
                if len(final_selection) >= top_n:
                    break
                if outfit in final_selection:
                    continue

                is_dress = any(item.category == ClothingCategory.DRESS for item in outfit.items)
                if is_dress and selected_dresses_count >= 1:
                    continue

                if _has_too_much_overlap(outfit, final_selection):
                    continue

                main_items = [
                    item.id for item in outfit.items 
                    if item.category in (ClothingCategory.DRESS, ClothingCategory.SHIRT, ClothingCategory.JACKET, ClothingCategory.PANTS, ClothingCategory.SKIRT, ClothingCategory.SHORTS)
                ]
                has_overlap = any(item_id in used_main_items for item_id in main_items)
                if has_overlap:
                    continue

                final_selection.append(outfit)
                if is_dress:
                    selected_dresses_count += 1
                else:
                    bottom_item = next((item for item in outfit.items if item.category in BOTTOM_CATEGORIES), None)
                    bottom_cat = bottom_item.category if bottom_item else None
                    if bottom_cat:
                        selected_bottom_cats[bottom_cat] = selected_bottom_cats.get(bottom_cat, 0) + 1
                for item_id in main_items:
                    used_main_items.add(item_id)

        # Pass 3: Relax main item overlap, strict at most 1 dress
        if len(final_selection) < top_n:
            for outfit in valid_candidates:
                if len(final_selection) >= top_n:
                    break
                if outfit in final_selection:
                    continue

                is_dress = any(item.category == ClothingCategory.DRESS for item in outfit.items)
                if is_dress and selected_dresses_count >= 1:
                    continue

                if _has_too_much_overlap(outfit, final_selection):
                    continue

                final_selection.append(outfit)
                if is_dress:
                    selected_dresses_count += 1
                else:
                    bottom_item = next((item for item in outfit.items if item.category in BOTTOM_CATEGORIES), None)
                    bottom_cat = bottom_item.category if bottom_item else None
                    if bottom_cat:
                        selected_bottom_cats[bottom_cat] = selected_bottom_cats.get(bottom_cat, 0) + 1

        # Pass 4: Fallback - absolute last resort. Fill up remaining slots with highest remaining scoring outfits.
        # Still strictly enforce the at-most-1-dress rule!
        if len(final_selection) < top_n:
            for outfit in valid_candidates:
                if len(final_selection) >= top_n:
                    break
                if outfit not in final_selection:
                    is_dress = any(item.category == ClothingCategory.DRESS for item in outfit.items)
                    if is_dress and selected_dresses_count >= 1:
                        continue
                    
                    if _has_too_much_overlap(outfit, final_selection):
                        continue

                    final_selection.append(outfit)
                    if is_dress:
                        selected_dresses_count += 1

        # Sort the diverse selection by compatibility score descending
        final_selection.sort(key=lambda o: o.score, reverse=True)

        return final_selection

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_outfit(self, items: List[ClothingItem]) -> Outfit:
        """Create an Outfit and compute its compatibility score."""
        outfit = Outfit(items)
        outfit.score = self.scorer.score(items)
        outfit.breakdown = self.scorer.score_breakdown(items)
        return outfit

    def _try_add_optionals(
        self,
        outfits: List[Outfit],
        jackets: List[ClothingItem],
        accessories: List[ClothingItem],
    ) -> List[Outfit]:
        """Try adding a jacket and/or an accessory if it improves the score.
        Adds at most ONE jacket and ONE accessory.
        """
        enhanced: List[Outfit] = []

        for outfit in outfits:
            best_outfit = outfit

            # Find the single best jacket
            best_jacket_outfit = best_outfit
            for jacket in jackets:
                if jacket in best_outfit.items:
                    continue
                candidate_items = best_outfit.items + [jacket]
                candidate = self._score_outfit(candidate_items)
                if candidate.score > best_jacket_outfit.score:
                    best_jacket_outfit = candidate
            
            best_outfit = best_jacket_outfit

            # Find the single best accessory
            best_acc_outfit = best_outfit
            for acc in accessories:
                if acc in best_outfit.items:
                    continue
                candidate_items = best_outfit.items + [acc]
                candidate = self._score_outfit(candidate_items)
                if candidate.score > best_acc_outfit.score:
                    best_acc_outfit = candidate
            
            best_outfit = best_acc_outfit
            enhanced.append(best_outfit)

        return enhanced

    def __repr__(self) -> str:
        return (
            f"OutfitGenerator(max_combos={self.max_combinations}, "
            f"min_score={self.min_score})"
        )
