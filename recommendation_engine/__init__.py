"""
Smart Wardrobe — Recommendation Engine
=======================================
Matches wardrobe items into outfits using cosine similarity,
K-Means clustering, and contextual filtering.
"""

# pyrefly: ignore [missing-import]
from .api import RecommendationAPI

__all__ = ["RecommendationAPI"]
__version__ = "1.0.0"
