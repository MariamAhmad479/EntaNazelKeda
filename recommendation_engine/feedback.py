"""
feedback.py
===========
Manages user feedback (accept / reject) on recommended outfits and
uses it to retrain compatibility scoring weights.

Workflow:
    1. User accepts or rejects an outfit.
    2. Feedback is logged with the outfit's sub-scores.
    3. When enough feedback accumulates, an XGBoost classifier is
       trained to predict accept/reject from sub-scores.
    4. Feature importances are mapped back to updated weights for the
       CompatibilityScorer.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from .compatibility import CompatibilityScorer


class FeedbackManager:
    """Collect user feedback and retrain scoring weights.

    Parameters
    ----------
    feedback_path : str
        Path to the JSON file where feedback is persisted.
    scorer : CompatibilityScorer
        The scorer whose weights will be updated on retrain.
    min_samples_to_retrain : int
        Minimum number of feedback entries required before retraining.
    """

    def __init__(
        self,
        feedback_path: str,
        scorer: CompatibilityScorer,
        min_samples_to_retrain: int = 10,
    ):
        self.feedback_path = feedback_path
        self.scorer = scorer
        self.min_samples_to_retrain = min_samples_to_retrain
        self._feedback: List[Dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load existing feedback from disk."""
        if os.path.exists(self.feedback_path):
            try:
                with open(self.feedback_path, "r", encoding="utf-8") as f:
                    self._feedback = json.load(f)
                if not isinstance(self._feedback, list):
                    self._feedback = []
            except (json.JSONDecodeError, IOError):
                self._feedback = []

    def _save(self) -> None:
        """Persist feedback to disk."""
        os.makedirs(os.path.dirname(self.feedback_path) or ".", exist_ok=True)
        with open(self.feedback_path, "w", encoding="utf-8") as f:
            json.dump(self._feedback, f, indent=2)

    # ------------------------------------------------------------------
    # Submit feedback
    # ------------------------------------------------------------------

    def submit(
        self,
        outfit_id: str,
        action: str,
        breakdown: Dict[str, float],
    ) -> None:
        """Record a user's accept/reject decision for an outfit.

        Parameters
        ----------
        outfit_id : str
            The outfit's unique id.
        action : str
            ``"accept"`` or ``"reject"``.
        breakdown : dict
            The sub-score breakdown from the CompatibilityScorer.
        """
        if action not in ("accept", "reject"):
            raise ValueError(f"Action must be 'accept' or 'reject', got {action!r}")

        # Always reload from disk first to stay in sync with other active instances
        self._load()

        entry = {
            "outfit_id": outfit_id,
            "action": action,
            "breakdown": breakdown,
            "timestamp": datetime.now().isoformat(),
        }
        self._feedback.append(entry)
        self._save()

    # ------------------------------------------------------------------
    # Retrain weights
    # ------------------------------------------------------------------

    def retrain(self) -> Optional[Dict[str, float]]:
        """Retrain scoring weights using XGBoost on accumulated feedback.

        Returns the new weights dict, or None if not enough data.
        """
        # Always reload from disk first to stay in sync
        self._load()
        if len(self._feedback) < self.min_samples_to_retrain:
            return None

        # Build training data
        feature_keys = ["color", "style", "formality", "similarity"]
        X = []
        y = []
        for entry in self._feedback:
            bd = entry.get("breakdown", {})
            row = [bd.get(k, 0.0) for k in feature_keys]
            X.append(row)
            y.append(1 if entry["action"] == "accept" else 0)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        # Check if we have both classes
        if len(set(y)) < 2:
            # Not enough diversity — can't train a meaningful model
            return None

        try:
            # py:ignore [missing-import]
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
            model.fit(X, y)

            # Map feature importances → new weights
            importances = model.feature_importances_
            total = importances.sum()
            if total > 0:
                new_weights = {
                    k: float(importances[i] / total)
                    for i, k in enumerate(feature_keys)
                }
            else:
                new_weights = {k: 0.25 for k in feature_keys}

        except ImportError:
            # Fall back to Random Forest if XGBoost is unavailable
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=3,
                random_state=42,
            )
            model.fit(X, y)
            importances = model.feature_importances_
            total = importances.sum()
            new_weights = {
                k: float(importances[i] / total)
                for i, k in enumerate(feature_keys)
            }

        # Apply new weights to the scorer
        self.scorer.set_weights(new_weights)
        return new_weights

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_feedback_count(self) -> int:
        """Return total number of feedback entries."""
        self._load()
        return len(self._feedback)

    def get_feedback_summary(self) -> Dict[str, int]:
        """Return counts of accepts and rejects."""
        self._load()
        accepts = sum(1 for f in self._feedback if f["action"] == "accept")
        rejects = len(self._feedback) - accepts
        return {"accept": accepts, "reject": rejects, "total": len(self._feedback)}

    def get_all_feedback(self) -> List[Dict]:
        """Return all raw feedback entries."""
        self._load()
        return list(self._feedback)

    def clear(self) -> None:
        """Clear all feedback (useful for testing)."""
        self._feedback = []
        self._save()

    def __repr__(self) -> str:
        return f"FeedbackManager(entries={len(self._feedback)})"
