"""
feature_encoder.py
==================
Encodes ClothingItem attributes into numerical feature vectors
suitable for cosine similarity, K-Means clustering, and ML models.

Encoding scheme (base dimensions ≈ 35):
    - Category         → one-hot   (8 dims)
    - Color RGB        → normalised (3 dims)
    - Pattern          → one-hot   (6 dims)
    - Style            → one-hot   (6 dims)
    - Occasions        → multi-hot (6 dims)
    - Seasons          → multi-hot (4 dims)
    - Warmth level     → scalar    (1 dim, normalised 0-1)
    - Formality level  → scalar    (1 dim, normalised 0-1)
    - CNN features     → pass-through (optional, variable dims)
"""

from __future__ import annotations

from typing import List

import numpy as np

from .data_models import (
    ClothingCategory,
    ClothingItem,
    Occasion,
    Pattern,
    Season,
    Style,
)


class FeatureEncoder:
    """Encode ClothingItem objects into fixed-length numerical vectors.

    The encoder produces deterministic, reproducible vectors whose
    dimensions are documented in the module docstring.
    """

    # Ordered member lists for consistent one-hot / multi-hot encoding
    CATEGORIES = list(ClothingCategory)
    PATTERNS = list(Pattern)
    STYLES = list(Style)
    OCCASIONS = list(Occasion)
    SEASONS = list(Season)

    # Base vector dimensionality (without CNN features)
    BASE_DIM = (
        len(CATEGORIES)   # 8
        + 3                # RGB
        + len(PATTERNS)    # 6
        + len(STYLES)      # 6
        + len(OCCASIONS)   # 6
        + len(SEASONS)     # 4
        + 1                # warmth
        + 1                # formality
    )  # = 35

    def __init__(self):
        self._cache = {}

    def encode(self, item: ClothingItem) -> np.ndarray:
        """Encode a single ClothingItem to a 1-D NumPy float array."""
        if item.id in self._cache:
            return self._cache[item.id]

        parts: List[np.ndarray] = []

        # 1. Category — one-hot
        cat_vec = np.zeros(len(self.CATEGORIES), dtype=np.float32)
        cat_vec[self.CATEGORIES.index(item.category)] = 1.0
        parts.append(cat_vec)

        # 2. Colour RGB — normalise to [0, 1]
        rgb = np.array(item.color_rgb, dtype=np.float32) / 255.0
        parts.append(rgb)

        # 3. Pattern — one-hot
        pat_vec = np.zeros(len(self.PATTERNS), dtype=np.float32)
        pat_vec[self.PATTERNS.index(item.pattern)] = 1.0
        parts.append(pat_vec)

        # 4. Style — one-hot
        sty_vec = np.zeros(len(self.STYLES), dtype=np.float32)
        sty_vec[self.STYLES.index(item.style)] = 1.0
        parts.append(sty_vec)

        # 5. Occasions — multi-hot
        occ_vec = np.zeros(len(self.OCCASIONS), dtype=np.float32)
        for occ in item.occasions:
            occ_vec[self.OCCASIONS.index(occ)] = 1.0
        parts.append(occ_vec)

        # 6. Seasons — multi-hot
        sea_vec = np.zeros(len(self.SEASONS), dtype=np.float32)
        for sea in item.seasons:
            sea_vec[self.SEASONS.index(sea)] = 1.0
        parts.append(sea_vec)

        # 7. Warmth level — normalise 1-5 → 0-1
        parts.append(np.array([(item.warmth_level - 1) / 4.0], dtype=np.float32))

        # 8. Formality level — normalise 1-5 → 0-1
        parts.append(np.array([(item.formality_level - 1) / 4.0], dtype=np.float32))

        # 9. CNN image features (optional)
        if item.image_features is not None:
            parts.append(np.array(item.image_features, dtype=np.float32))

        vec = np.concatenate(parts)
        self._cache[item.id] = vec
        return vec

    def encode_many(self, items: List[ClothingItem]) -> np.ndarray:
        """Encode a list of items into a 2-D array (N × D).

        If items have inconsistent CNN feature sizes, this will raise
        a ValueError due to shape mismatch.
        """
        vectors = [self.encode(item) for item in items]
        return np.vstack(vectors)

    def get_feature_dim(self, has_cnn_features: bool = False,
                        cnn_dim: int = 0) -> int:
        """Return the expected feature vector dimensionality."""
        return self.BASE_DIM + (cnn_dim if has_cnn_features else 0)

    def __repr__(self) -> str:
        return f"FeatureEncoder(base_dim={self.BASE_DIM})"
