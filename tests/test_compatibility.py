"""Tests for the CompatibilityScorer module."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recommendation_engine.data_models import (
    ClothingCategory, Occasion, Season, Style, Pattern, ClothingItem,
)
from recommendation_engine.compatibility import CompatibilityScorer


def _make(name, cat, rgb, style, formality=3):
    return ClothingItem(
        name=name, category=cat,
        color_rgb=rgb, color_name="test",
        pattern=Pattern.SOLID, style=style,
        occasions=[Occasion.CASUAL], seasons=[Season.SUMMER],
        warmth_level=2, formality_level=formality,
    )


@pytest.fixture
def scorer():
    return CompatibilityScorer()


def test_score_returns_float(scorer):
    items = [
        _make("A", ClothingCategory.SHIRT, (255, 255, 255), Style.CLASSIC),
        _make("B", ClothingCategory.PANTS, (0, 0, 0), Style.CLASSIC),
    ]
    s = scorer.score(items)
    assert isinstance(s, float)


def test_score_in_range(scorer):
    items = [
        _make("A", ClothingCategory.SHIRT, (255, 0, 0), Style.CLASSIC),
        _make("B", ClothingCategory.PANTS, (0, 0, 255), Style.MINIMALIST, formality=1),
        _make("C", ClothingCategory.SHOES, (0, 255, 0), Style.ATHLETIC, formality=5),
    ]
    s = scorer.score(items)
    assert 0.0 <= s <= 1.0


def test_single_item_score(scorer):
    items = [_make("Solo", ClothingCategory.SHIRT, (100, 100, 100), Style.CLASSIC)]
    assert scorer.score(items) == 1.0


def test_matching_styles_score_higher(scorer):
    matching = [
        _make("A", ClothingCategory.SHIRT, (200, 200, 200), Style.CLASSIC, 3),
        _make("B", ClothingCategory.PANTS, (180, 180, 180), Style.CLASSIC, 3),
    ]
    mismatched = [
        _make("A", ClothingCategory.SHIRT, (200, 200, 200), Style.CLASSIC, 3),
        _make("B", ClothingCategory.PANTS, (180, 180, 180), Style.ATHLETIC, 1),
    ]
    assert scorer.score(matching) > scorer.score(mismatched)


def test_breakdown_has_all_keys(scorer):
    items = [
        _make("A", ClothingCategory.SHIRT, (255, 255, 255), Style.CLASSIC),
        _make("B", ClothingCategory.PANTS, (0, 0, 0), Style.CLASSIC),
    ]
    bd = scorer.score_breakdown(items)
    assert set(bd.keys()) == {"color", "style", "formality", "similarity"}


def test_set_weights(scorer):
    scorer.set_weights({"color": 1.0, "style": 0.0, "formality": 0.0, "similarity": 0.0})
    w = scorer.get_weights()
    assert w["color"] == 1.0
    assert w["style"] == 0.0


def test_neutral_colors_high_harmony(scorer):
    """White + grey (both low saturation) should have high color harmony."""
    items = [
        _make("White", ClothingCategory.SHIRT, (240, 240, 240), Style.CLASSIC),
        _make("Grey", ClothingCategory.PANTS, (130, 130, 130), Style.CLASSIC),
    ]
    bd = scorer.score_breakdown(items)
    assert bd["color"] >= 0.9
