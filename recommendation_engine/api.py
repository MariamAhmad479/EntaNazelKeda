"""
api.py — Public API for the Smart Wardrobe Recommendation Engine.
"""
from __future__ import annotations
import os, sys
from typing import Dict, List, Optional
from .clustering import ClothingClusterer
from .compatibility import CompatibilityScorer
from .context_filter import ContextFilter
from .data_models import ClothingItem
from .feedback import FeedbackManager
from .outfit_generator import OutfitGenerator
from .wardrobe_manager import WardrobeManager

class RecommendationAPI:
    def __init__(self, wardrobe_path: str, feedback_path: Optional[str] = None):
        self.wardrobe_path = wardrobe_path
        if feedback_path is None:
            base_dir = os.path.dirname(wardrobe_path) or "."
            feedback_path = os.path.join(base_dir, "feedback_log.json")
        self.feedback_path = feedback_path
        self._wardrobe = WardrobeManager(wardrobe_path)
        self._scorer = CompatibilityScorer()
        self._generator = OutfitGenerator(scorer=self._scorer)
        self._filter = ContextFilter()
        self._feedback = FeedbackManager(feedback_path, self._scorer)
        self._clusterer = ClothingClusterer()
        if self._wardrobe.size > 0:
            self._clusterer.fit(self._wardrobe.get_all_items())

    def add_item(self, item_dict: dict) -> str:
        item_id = self._wardrobe.add_item_from_dict(item_dict)
        self._clusterer.fit(self._wardrobe.get_all_items())
        return item_id

    def remove_item(self, item_id: str) -> bool:
        result = self._wardrobe.remove_item(item_id)
        if result and self._wardrobe.size > 0:
            self._clusterer.fit(self._wardrobe.get_all_items())
        return result

    def get_wardrobe(self) -> List[dict]:
        return [item.to_dict() for item in self._wardrobe.get_all_items()]

    def get_wardrobe_summary(self) -> dict:
        return self._wardrobe.summary()

    def get_outfits(self, occasion=None, weather=None, style=None, top_n=5) -> List[dict]:
        items = self._wardrobe.get_all_items()
        if not items:
            return []
        filtered = self._filter.filter_items(items, occasion=occasion, weather=weather, style=style)
        outfits = self._generator.generate(filtered, top_n=top_n)
        return [o.to_dict() for o in outfits]

    def submit_feedback(self, outfit_id: str, action: str) -> None:
        breakdown = self._scorer.get_weights()
        self._feedback.submit(outfit_id, action, breakdown)

    def retrain(self) -> Optional[Dict[str, float]]:
        return self._feedback.retrain()

    def get_feedback_summary(self) -> dict:
        return self._feedback.get_feedback_summary()

    def get_clusters(self) -> dict:
        all_clusters = self._clusterer.get_all_clusters()
        return {str(cid): [item.to_dict() for item in members] for cid, members in all_clusters.items()}

    def get_scoring_weights(self) -> Dict[str, float]:
        return self._scorer.get_weights()

    def __repr__(self) -> str:
        return f"RecommendationAPI(wardrobe={self._wardrobe.size} items, feedback={self._feedback.get_feedback_count()} entries)"

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    wardrobe_file = os.path.join(data_dir, "sample_wardrobe.json")
    if not os.path.exists(wardrobe_file):
        print(f"Wardrobe file not found: {wardrobe_file}")
        sys.exit(1)
    api = RecommendationAPI(wardrobe_file)
    print(f"Loaded: {api}")
    print(f"Wardrobe summary: {api.get_wardrobe_summary()}")
    print(f"Scoring weights: {api.get_scoring_weights()}")
    print("\n" + "=" * 60)
    print("TOP 5 OUTFITS (no filter)")
    print("=" * 60)
    for i, o in enumerate(api.get_outfits(top_n=5), 1):
        print(f"  #{i} Score: {o['score']:.4f} — {o['summary']}")
    print("\n" + "=" * 60)
    print("TOP 3 FORMAL OUTFITS")
    print("=" * 60)
    for i, o in enumerate(api.get_outfits(occasion="formal", top_n=3), 1):
        print(f"  #{i} Score: {o['score']:.4f} — {o['summary']}")
    print("\n" + "=" * 60)
    print("WARDROBE CLUSTERS")
    print("=" * 60)
    for cid, members in api.get_clusters().items():
        print(f"  Cluster {cid}: {', '.join(m['name'] for m in members)}")
