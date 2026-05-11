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
            combos = list(product(tops, bottoms, shoes))
            # Limit combinatorial explosion
            if len(combos) > self.max_combinations:
                import random
                random.seed(42)
                combos = random.sample(combos, self.max_combinations)

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
            candidates = self._try_add_optionals(
                candidates[:top_n * 2], jackets, accessories
            )
            candidates.sort(key=lambda o: o.score, reverse=True)

        return candidates[:top_n]

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
        """Try adding a jacket or accessory if it improves the score."""
        enhanced: List[Outfit] = []

        for outfit in outfits:
            best = outfit

            # Try each jacket
            for jacket in jackets:
                if jacket in outfit.items:
                    continue
                candidate_items = outfit.items + [jacket]
                candidate = self._score_outfit(candidate_items)
                if candidate.score > best.score:
                    best = candidate

            # Try each accessory on the current best
            for acc in accessories:
                if acc in best.items:
                    continue
                candidate_items = best.items + [acc]
                candidate = self._score_outfit(candidate_items)
                if candidate.score > best.score:
                    best = candidate

            enhanced.append(best)

        return enhanced

    def __repr__(self) -> str:
        return (
            f"OutfitGenerator(max_combos={self.max_combinations}, "
            f"min_score={self.min_score})"
        )
