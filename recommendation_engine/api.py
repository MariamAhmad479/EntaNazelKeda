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

# Vision model is optional — only imported when used so TF is not mandatory
try:
    from vision import VisionModel as _VisionModel
    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False

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
        self._vision = None  # lazy – only initialised when vision methods are called
        if self._wardrobe.size > 0:
            self._clusterer.fit(self._wardrobe.get_all_items())

    # ------------------------------------------------------------------
    # Alternative constructor — build from Person 1's vision data
    # ------------------------------------------------------------------

    @classmethod
    def from_vision_data(
        cls,
        vision_dir: str,
        wardrobe_path: str = "data/vision_wardrobe.json",
        feedback_path: Optional[str] = None,
    ) -> "RecommendationAPI":
        """Create a RecommendationAPI from Person 1's vision artifacts.

        Parameters
        ----------
        vision_dir : str
            Path to ``vision/saved/`` containing the CSV and NPY files.
        wardrobe_path : str
            Path where the converted wardrobe JSON will be saved.
        feedback_path : str or None
            Path to the feedback log file.

        Returns
        -------
        RecommendationAPI
            Fully initialised API backed by real vision data.
        """
        # Build the wardrobe from vision data (also saves as JSON)
        wardrobe = WardrobeManager.from_vision_data(vision_dir, wardrobe_path)

        # Now create the API using the saved JSON path
        instance = cls.__new__(cls)
        instance.wardrobe_path = wardrobe_path
        if feedback_path is None:
            base_dir = os.path.dirname(wardrobe_path) or "."
            feedback_path = os.path.join(base_dir, "feedback_log.json")
        instance.feedback_path = feedback_path
        instance._wardrobe = wardrobe
        instance._scorer = CompatibilityScorer()
        instance._generator = OutfitGenerator(scorer=instance._scorer)
        instance._filter = ContextFilter()
        instance._feedback = FeedbackManager(feedback_path, instance._scorer)
        instance._clusterer = ClothingClusterer()
        instance._vision = None
        if instance._wardrobe.size > 0:
            instance._clusterer.fit(instance._wardrobe.get_all_items())
        return instance

    # ------------------------------------------------------------------
    # Vision model helpers
    # ------------------------------------------------------------------

    def _get_vision(self):
        """Return a cached VisionModel instance (lazy init)."""
        if self._vision is None:
            if not _VISION_AVAILABLE:
                raise RuntimeError(
                    "The 'vision' package is not available. "
                    "Make sure vision/vision_model.py is on the Python path."
                )
            self._vision = _VisionModel()
        return self._vision

    def reload_wardrobe(self, wardrobe_path: str):
        """Switch the API to a new wardrobe JSON file and re-index items."""
        self.wardrobe_path = wardrobe_path
        self._wardrobe = WardrobeManager(wardrobe_path)
        if self._wardrobe.size > 0:
            self._clusterer.fit(self._wardrobe.get_all_items())
        else:
            self._clusterer = ClothingClusterer()  # reset if empty
        print(f"[API] Switched wardrobe to: {os.path.basename(wardrobe_path)}")

    def analyze_image(self, img_path: str) -> Dict:
        """Run the vision model on an image and return analysis results.

        Parameters
        ----------
        img_path : str
            Path to the garment image file.

        Returns
        -------
        dict with keys:
            ``category``       – ClothingCategory value (e.g. "shirt")
            ``kaggle_label``   – Fine-grained Kaggle label (e.g. "Tshirts")
            ``confidence``     – Classifier probability (float)
            ``image_features`` – 2048-D ResNet50 feature vector (list[float])

        Example
        -------
        >>> api = RecommendationAPI("data/sample_wardrobe.json")
        >>> result = api.analyze_image("images/12345.jpg")
        >>> print(result["category"], result["confidence"])
        shirt 0.9231
        """
        return self._get_vision().analyze(img_path)

    def add_item_from_image(self, img_path: str, item_dict: dict) -> str:
        """Add a wardrobe item, auto-filling category and image features from vision.

        The vision model analyses the image and:
          - Sets ``item_dict["category"]`` if not already provided.
          - Sets ``item_dict["image_features"]`` to the ResNet50 embedding.
          - Sets ``item_dict["image_path"]`` to ``img_path``.

        All other fields (name, color_rgb, color_name, pattern, style,
        occasions, seasons, warmth_level, formality_level) must be supplied
        by the caller.

        Parameters
        ----------
        img_path : str
            Path to the garment image.
        item_dict : dict
            Partial ClothingItem dict. ``category`` and ``image_features``
            will be overwritten by the vision model results.

        Returns
        -------
        str
            The new item's id.

        Example
        -------
        >>> item_id = api.add_item_from_image(
        ...     "images/12345.jpg",
        ...     {
        ...         "name": "Blue Oxford",
        ...         "color_rgb": [70, 130, 180],
        ...         "color_name": "steel blue",
        ...         "pattern": "solid",
        ...         "style": "classic",
        ...         "occasions": ["casual", "business"],
        ...         "seasons": ["spring", "autumn"],
        ...         "warmth_level": 2,
        ...         "formality_level": 3,
        ...     }
        ... )
        """
        vision_result = self._get_vision().analyze(img_path)

        # Only overwrite category if caller did not provide one
        if "category" not in item_dict:
            item_dict["category"] = vision_result["category"]

        # Always attach the ResNet50 embedding and image path
        item_dict["image_features"] = vision_result["image_features"]
        item_dict["image_path"] = img_path

        return self.add_item(item_dict)

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

    def get_outfits(self, occasion=None, weather=None, style=None, gender=None, color=None, piece=None, top_n=5) -> List[dict]:
        items = self._wardrobe.get_all_items()
        if not items:
            return []
        filtered = self._filter.filter_items(items, occasion=occasion, weather=weather, style=style, gender=gender, color=color, piece=piece)
        
        if style is not None:
            # Choose top 10 compatible outfits
            outfits = self._generator.generate(filtered, top_n=10)
            if not outfits and filtered:
                outfits = self._generator.generate(filtered, top_n=10, min_score=0.20)
            
            # Randomly select 3 outfits from the top 10
            import random
            if len(outfits) > 3:
                outfits = random.sample(outfits, 3)
            # Sort by compatibility score descending to keep them ordered nicely
            outfits.sort(key=lambda o: o.score, reverse=True)
        else:
            outfits = self._generator.generate(filtered, top_n=top_n)
            if not outfits and filtered:
                outfits = self._generator.generate(filtered, top_n=top_n, min_score=0.20)
            
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
