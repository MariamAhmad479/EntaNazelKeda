"""
clustering.py
=============
Groups wardrobe items into style/similarity clusters using K-Means.

Provides:
- Automatic cluster count selection via the Elbow Method
- Cluster assignment for each item
- Retrieval of cluster members
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

from .data_models import ClothingItem
from .feature_encoder import FeatureEncoder


class ClothingClusterer:
    """Cluster wardrobe items using K-Means on their feature vectors.

    Parameters
    ----------
    n_clusters : int or None
        Number of clusters. If None, the optimal k is auto-selected
        via the Elbow Method (testing k=2..max_k).
    max_k : int
        Maximum k to test when auto-selecting (default 10).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        max_k: int = 10,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.max_k = max_k
        self.random_state = random_state

        self._encoder = FeatureEncoder()
        self._kmeans: Optional[KMeans] = None
        self._items: List[ClothingItem] = []
        self._labels: Optional[np.ndarray] = None
        self._features: Optional[np.ndarray] = None
        self._inertias: List[Tuple[int, float]] = []

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, items: List[ClothingItem]) -> "ClothingClusterer":
        """Fit K-Means on the wardrobe items.

        If ``n_clusters`` was not specified, the Elbow Method is used
        to pick an appropriate k.
        """
        self._items = list(items)
        self._features = self._encoder.encode_many(self._items)

        n_samples = len(self._items)
        if n_samples < 2:
            # Not enough items to cluster
            self._labels = np.zeros(n_samples, dtype=int)
            self.n_clusters = 1
            return self

        if self.n_clusters is None:
            self.n_clusters = self._find_optimal_k(self._features)

        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self._labels = self._kmeans.fit_predict(self._features)
        return self

    def _find_optimal_k(self, features: np.ndarray) -> int:
        """Use the Elbow Method (inertia) to pick k.

        Returns the k at the point of maximum curvature, or a sensible
        default if the data is too small.
        """
        max_k = min(self.max_k, len(features))
        if max_k < 2:
            return 1

        self._inertias = []
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(features)
            self._inertias.append((k, km.inertia_))

        # Simple heuristic: find the k where the *decrease* in inertia
        # drops below 50 % of the previous decrease (the "elbow").
        if len(self._inertias) < 2:
            return 2

        prev_drop = None
        for i in range(1, len(self._inertias)):
            drop = self._inertias[i - 1][1] - self._inertias[i][1]
            if prev_drop is not None and drop < 0.5 * prev_drop:
                return self._inertias[i - 1][0]
            prev_drop = drop

        # Default: pick k = 3 if nothing stands out
        return min(3, max_k)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_cluster(self, item_id: str) -> int:
        """Return the cluster label for a given item id."""
        for idx, item in enumerate(self._items):
            if item.id == item_id:
                return int(self._labels[idx])
        raise KeyError(f"Item {item_id!r} not found in clustered items.")

    def get_cluster_members(self, cluster_id: int) -> List[ClothingItem]:
        """Return all items belonging to a cluster."""
        return [
            item
            for idx, item in enumerate(self._items)
            if self._labels[idx] == cluster_id
        ]

    def get_all_clusters(self) -> Dict[int, List[ClothingItem]]:
        """Return a dict mapping cluster_id → list of items."""
        clusters: Dict[int, List[ClothingItem]] = {}
        for idx, item in enumerate(self._items):
            cid = int(self._labels[idx])
            clusters.setdefault(cid, []).append(item)
        return clusters

    def get_inertias(self) -> List[Tuple[int, float]]:
        """Return the (k, inertia) pairs from the Elbow Method search."""
        return list(self._inertias)

    @property
    def labels(self) -> Optional[np.ndarray]:
        """Cluster labels for all items (in the order they were fitted)."""
        return self._labels

    def __repr__(self) -> str:
        n = len(self._items)
        k = self.n_clusters
        return f"ClothingClusterer(n_items={n}, k={k})"
