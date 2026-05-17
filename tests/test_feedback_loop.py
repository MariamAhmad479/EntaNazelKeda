import pytest
import sys, os
import tempfile
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recommendation_engine.api import RecommendationAPI

def test_feedback_logging_and_retraining():
    # Setup temporary files for testing wardrobe and feedback
    with tempfile.TemporaryDirectory() as tmpdir:
        wardrobe_path = os.path.join(tmpdir, "wardrobe.json")
        feedback_path = os.path.join(tmpdir, "feedback_log.json")
        
        # Simple dummy wardrobe
        dummy_items = [
            {
                "id": "item1",
                "name": "White Shirt",
                "category": "shirt",
                "color_rgb": [255, 255, 255],
                "color_name": "white",
                "pattern": "solid",
                "style": "classic",
                "occasions": ["casual"],
                "seasons": ["summer"],
                "warmth_level": 2,
                "formality_level": 3,
            },
            {
                "id": "item2",
                "name": "Blue Jeans",
                "category": "pants",
                "color_rgb": [0, 0, 255],
                "color_name": "blue",
                "pattern": "solid",
                "style": "classic",
                "occasions": ["casual"],
                "seasons": ["summer"],
                "warmth_level": 2,
                "formality_level": 2,
            },
            {
                "id": "item3",
                "name": "Black Shoes",
                "category": "shoes",
                "color_rgb": [0, 0, 0],
                "color_name": "black",
                "pattern": "solid",
                "style": "classic",
                "occasions": ["casual"],
                "seasons": ["summer"],
                "warmth_level": 2,
                "formality_level": 3,
            }
        ]
        
        with open(wardrobe_path, "w", encoding="utf-8") as f:
            json.dump(dummy_items, f)
            
        api = RecommendationAPI(wardrobe_path=wardrobe_path, feedback_path=feedback_path)
        
        # 1. Verify feedback summary starts empty
        summary = api.get_feedback_summary()
        assert summary.get("accept", 0) == 0
        assert summary.get("reject", 0) == 0
        assert summary.get("total", 0) == 0
        
        # 2. Log 3 accepts and 2 rejects
        api.submit_feedback("outfit_a", "accept")
        api.submit_feedback("outfit_b", "accept")
        api.submit_feedback("outfit_c", "accept")
        api.submit_feedback("outfit_d", "reject")
        api.submit_feedback("outfit_e", "reject")
        
        summary = api.get_feedback_summary()
        assert summary.get("accept", 0) == 3
        assert summary.get("reject", 0) == 2
        assert summary.get("total", 0) == 5
        
        # 3. Log enough samples to trigger retraining (>= 10 total)
        for i in range(5):
            api.submit_feedback(f"outfit_accept_{i}", "accept")
            api.submit_feedback(f"outfit_reject_{i}", "reject")
            
        summary = api.get_feedback_summary()
        assert summary.get("total", 0) == 15
        
        # 4. Trigger Retraining
        weights = api.retrain()
        # Even if it falls back to sklearn, it must return valid weights dict
        assert weights is not None
        assert "color" in weights
        assert "style" in weights
        assert "formality" in weights
        assert "similarity" in weights
        
        # 5. Check if updated weights are visible
        active_weights = api.get_scoring_weights()
        assert active_weights == weights
