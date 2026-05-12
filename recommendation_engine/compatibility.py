"""
compatibility.py
================
Scores how well a set of clothing items work together as an outfit.

Uses a weighted combination of:
    1. Color harmony  — complementary / analogous hue analysis (HSV)
    2. Style coherence — fraction of items sharing the same Style
    3. Formality match — low std-deviation of formality levels
    4. Feature similarity — avg pairwise cosine similarity of vectors
"""

from __future__ import annotations

import colorsys
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .data_models import ClothingItem
from .feature_encoder import FeatureEncoder


class CompatibilityScorer:
    """Score outfit compatibility on a 0-1 scale.

    Parameters
    ----------
    weights : dict or None
        Custom weights for each sub-score. Keys:
        ``color``, ``style``, ``formality``, ``similarity``.
        If None, sensible defaults are used.
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "color": 0.25,
        "style": 0.25,
        "formality": 0.25,
        "similarity": 0.25,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

        # Normalise so weights sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        self._encoder = FeatureEncoder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, items: List[ClothingItem]) -> float:
        """Return an overall compatibility score in [0, 1]."""
        if len(items) < 2:
            return 1.0  # a single item is trivially compatible

        sub = self.score_breakdown(items)
        return sum(self.weights[k] * sub[k] for k in self.weights)

    def score_breakdown(self, items: List[ClothingItem]) -> Dict[str, float]:
        """Return individual sub-scores for an outfit."""
        return {
            "color": self._color_harmony(items),
            "style": self._style_coherence(items),
            "formality": self._formality_match(items),
            "similarity": self._feature_similarity(items),
        }

    # ------------------------------------------------------------------
    # Sub-score: color harmony
    # ------------------------------------------------------------------

    @staticmethod
    def _rgb_to_hsv(rgb: tuple) -> tuple:
        """Convert (R, G, B) in 0-255 to (H, S, V) with H in degrees."""
        r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h * 360, s, v

    def _color_harmony(self, items: List[ClothingItem]) -> float:
        """Score based on how well item hues harmonise.

        Complementary (Δhue ≈ 180°) and analogous (Δhue ≈ 30°)
        combinations are rewarded.  Clash (Δhue ≈ 90° / 270°) is
        penalised.
        """
        hsvs = [self._rgb_to_hsv(item.color_rgb) for item in items]

        if len(hsvs) < 2:
            return 1.0

        pair_scores = []
        for (h1, s1, _), (h2, s2, _) in combinations(hsvs, 2):
            # If both colours are low-saturation (neutrals), they always match
            if s1 < 0.15 and s2 < 0.15:
                pair_scores.append(1.0)
                continue

            # If one is neutral, it pairs well with anything
            if s1 < 0.15 or s2 < 0.15:
                pair_scores.append(0.9)
                continue

            delta = abs(h1 - h2) % 360
            if delta > 180:
                delta = 360 - delta

            # Score based on hue difference
            if delta <= 30:
                # Analogous — great harmony
                pair_scores.append(1.0)
            elif delta <= 60:
                pair_scores.append(0.85)
            elif 150 <= delta <= 180:
                # Complementary — bold but harmonious
                pair_scores.append(0.9)
            elif 120 <= delta < 150:
                # Triadic-ish
                pair_scores.append(0.75)
            else:
                # Less harmonious
                pair_scores.append(0.5)

        return float(np.mean(pair_scores)) if pair_scores else 1.0

    # ------------------------------------------------------------------
    # Sub-score: style coherence
    # ------------------------------------------------------------------

    @staticmethod
    def _style_coherence(items: List[ClothingItem]) -> float:
        """Fraction of items that share the most-common style."""
        if not items:
            return 1.0
        from collections import Counter
        styles = [item.style for item in items]
        most_common_count = Counter(styles).most_common(1)[0][1]
        return most_common_count / len(items)

    # ------------------------------------------------------------------
    # Sub-score: formality match
    # ------------------------------------------------------------------

    @staticmethod
    def _formality_match(items: List[ClothingItem]) -> float:
        """Score inversely proportional to formality spread.

        If all items share the same formality level → 1.0.
        Max possible std for levels 1-5 is 2.0, so we normalise by that.
        """
        if len(items) < 2:
            return 1.0
        levels = np.array([item.formality_level for item in items], dtype=float)
        std = float(np.std(levels))
        # Map std ∈ [0, 2] → score ∈ [1, 0]
        return max(0.0, 1.0 - std / 2.0)

    # ------------------------------------------------------------------
    # Sub-score: feature vector similarity
    # ------------------------------------------------------------------

    def _feature_similarity(self, items: List[ClothingItem]) -> float:
        """Average pairwise cosine similarity of encoded feature vectors."""
        if len(items) < 2:
            return 1.0
        vectors = self._encoder.encode_many(items)
        sim_matrix = cosine_similarity(vectors)

        # Extract upper triangle (exclude self-similarity on the diagonal)
        n = sim_matrix.shape[0]
        upper = [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
        avg_sim = float(np.mean(upper))

        # Cosine sim for these vectors is typically 0.3–0.9, rescale to 0-1
        return float(np.clip(avg_sim, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update scoring weights (will be normalised)."""
        self.weights.update(weights)
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def get_weights(self) -> Dict[str, float]:
        """Return current scoring weights."""
        return dict(self.weights)

    def __repr__(self) -> str:
        w = ", ".join(f"{k}={v:.2f}" for k, v in self.weights.items())
        return f"CompatibilityScorer({w})"
