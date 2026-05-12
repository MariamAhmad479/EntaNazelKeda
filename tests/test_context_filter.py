"""Tests for the ContextFilter module."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recommendation_engine.data_models import (
    ClothingCategory, Occasion, Season, Style, Pattern, ClothingItem,
)
from recommendation_engine.context_filter import ContextFilter


def _make(name, occasions, seasons, style, warmth=2, formality=3):
    return ClothingItem(
        name=name, category=ClothingCategory.SHIRT,
        color_rgb=(128, 128, 128), color_name="grey",
        pattern=Pattern.SOLID, style=style,
        occasions=occasions, seasons=seasons,
        warmth_level=warmth, formality_level=formality,
    )


@pytest.fixture
def cf():
    return ContextFilter()


@pytest.fixture
def items():
    return [
        _make("Formal Classic", [Occasion.FORMAL, Occasion.BUSINESS],
              [Season.AUTUMN, Season.WINTER], Style.CLASSIC, warmth=3, formality=5),
        _make("Casual Summer", [Occasion.CASUAL],
              [Season.SUMMER], Style.STREETWEAR, warmth=1, formality=1),
        _make("Sport Item", [Occasion.SPORT, Occasion.CASUAL],
              [Season.SPRING, Season.SUMMER], Style.ATHLETIC, warmth=1, formality=1),
        _make("Party Classic", [Occasion.PARTY, Occasion.FORMAL],
              [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER],
              Style.CLASSIC, warmth=2, formality=4),
    ]


def test_filter_by_occasion(cf, items):
    result = cf.filter_items(items, occasion="formal")
    assert len(result) == 2  # Formal Classic + Party Classic
    for item in result:
        assert Occasion.FORMAL in item.occasions


def test_filter_by_style(cf, items):
    result = cf.filter_items(items, style="classic")
    assert all(item.style == Style.CLASSIC for item in result)


def test_filter_by_weather_hot(cf, items):
    result = cf.filter_items(items, weather={"temperature": 30, "condition": "sunny"})
    # Should prefer warmth 1-2 items suited for summer
    assert len(result) >= 1


def test_filter_by_weather_cold(cf, items):
    result = cf.filter_items(items, weather={"temperature": 0, "condition": "snowy"})
    # Should prefer warmth 4-5 or tolerate 3
    assert len(result) >= 1


def test_combined_filters(cf, items):
    result = cf.filter_items(items, occasion="casual", style="athletic")
    assert len(result) >= 1


def test_unknown_occasion_returns_all(cf, items):
    result = cf.filter_items(items, occasion="wedding")
    assert len(result) == len(items)


def test_unknown_style_returns_all(cf, items):
    result = cf.filter_items(items, style="gothic")
    assert len(result) == len(items)


def test_no_filter_returns_all(cf, items):
    result = cf.filter_items(items)
    assert len(result) == len(items)
