"""Tests for the OutfitGenerator module."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recommendation_engine.data_models import (
    ClothingCategory, Occasion, Season, Style, Pattern, ClothingItem,
)
from recommendation_engine.outfit_generator import OutfitGenerator, Outfit


def _make(name, cat, rgb=(128, 128, 128), style=Style.CLASSIC, formality=3):
    return ClothingItem(
        name=name, category=cat,
        color_rgb=rgb, color_name="test",
        pattern=Pattern.SOLID, style=style,
        occasions=[Occasion.CASUAL], seasons=[Season.SUMMER],
        warmth_level=2, formality_level=formality,
    )


@pytest.fixture
def wardrobe():
    return [
        _make("Shirt 1", ClothingCategory.SHIRT),
        _make("Shirt 2", ClothingCategory.SHIRT),
        _make("Pants 1", ClothingCategory.PANTS),
        _make("Pants 2", ClothingCategory.PANTS),
        _make("Shoes 1", ClothingCategory.SHOES),
        _make("Dress 1", ClothingCategory.DRESS),
    ]


@pytest.fixture
def generator():
    return OutfitGenerator(min_score=0.0)


def test_generate_returns_list(generator, wardrobe):
    outfits = generator.generate(wardrobe, top_n=5)
    assert isinstance(outfits, list)
    assert all(isinstance(o, Outfit) for o in outfits)


def test_outfits_have_shoes(generator, wardrobe):
    outfits = generator.generate(wardrobe, top_n=10)
    for outfit in outfits:
        cats = [i.category for i in outfit.items]
        assert ClothingCategory.SHOES in cats


def test_standard_outfit_has_top_and_bottom(generator, wardrobe):
    outfits = generator.generate(wardrobe, top_n=10, include_optional=False)
    for outfit in outfits:
        cats = set(i.category for i in outfit.items)
        has_dress = ClothingCategory.DRESS in cats
        has_top = ClothingCategory.SHIRT in cats
        has_bottom = cats.intersection({ClothingCategory.PANTS, ClothingCategory.SKIRT, ClothingCategory.SHORTS})
        assert has_dress or (has_top and has_bottom)


def test_sorted_by_score(generator, wardrobe):
    outfits = generator.generate(wardrobe, top_n=5)
    scores = [o.score for o in outfits]
    assert scores == sorted(scores, reverse=True)


def test_top_n_limit(generator, wardrobe):
    outfits = generator.generate(wardrobe, top_n=2)
    assert len(outfits) <= 2


def test_outfit_to_dict(generator, wardrobe):
    outfits = generator.generate(wardrobe, top_n=1)
    if outfits:
        d = outfits[0].to_dict()
        assert "outfit_id" in d
        assert "score" in d
        assert "items" in d
        assert "summary" in d


def test_empty_wardrobe():
    gen = OutfitGenerator(min_score=0.0)
    outfits = gen.generate([], top_n=5)
    assert outfits == []


def test_no_shoes_no_outfits():
    items = [
        _make("Shirt", ClothingCategory.SHIRT),
        _make("Pants", ClothingCategory.PANTS),
    ]
    gen = OutfitGenerator(min_score=0.0)
    outfits = gen.generate(items, top_n=5)
    assert outfits == []
