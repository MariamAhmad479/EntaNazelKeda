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
    ) -> List[Outfit]:
        """Generate the top-N outfits from the given items.

        Returns outfits sorted by descending compatibility score.
        """
        # Partition items by slot
        tops = [i for i in items if i.category in TOP_CATEGORIES and i.category != ClothingCategory.JACKET]
        bottoms = [i for i in items if i.category in BOTTOM_CATEGORIES]
        dresses = [i for i in items if i.category in FULL_BODY_CATEGORIES]
        shoes = [i for i in items if i.category in FOOTWEAR_CATEGORIES]
        jackets = [i for i in items if i.category == ClothingCategory.JACKET]
        accessories = [i for i in items if i.category == ClothingCategory.ACCESSORY]

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
                    if outfit.score >= self.min_score:
                        candidates.append(outfit)
            else:
                # Small enough to materialize
                combos = list(product(tops, bottoms, shoes))
                for top, bottom, shoe in combos:
                    base_items = [top, bottom, shoe]
                    outfit = self._score_outfit(base_items)
                    if outfit.score >= self.min_score:
                        candidates.append(outfit)

        # --- Dress outfits: dress + shoes ---
        if dresses and shoes:
            for dress, shoe in product(dresses, shoes):
                base_items = [dress, shoe]
                outfit = self._score_outfit(base_items)
                if outfit.score >= self.min_score:
                    candidates.append(outfit)

        # Sort by score descending
        candidates.sort(key=lambda o: o.score, reverse=True)

        # Optionally try adding jackets / accessories to top outfits
        if include_optional and candidates:
            # Only consider the top few jackets/accessories to save time
            candidates = self._try_add_optionals(
                candidates[:top_n * 2], jackets[:30], accessories[:30]
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

        # Introduce variety by randomly sampling from a larger pool of top candidates
        pool_size = top_n * 3
        top_pool = valid_candidates[:pool_size]
        
        import random
        if len(top_pool) > top_n:
            final_selection = random.sample(top_pool, top_n)
            final_selection.sort(key=lambda o: o.score, reverse=True)
        else:
            final_selection = top_pool

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
