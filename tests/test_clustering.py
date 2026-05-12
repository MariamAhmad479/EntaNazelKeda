"""Tests for the ClothingClusterer module."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recommendation_engine.data_models import (
    ClothingCategory, Occasion, Season, Style, Pattern, ClothingItem,
)
from recommendation_engine.clustering import ClothingClusterer


def _make_item(name, category, color, style, warmth=2, formality=2):
    return ClothingItem(
        name=name, category=category,
        color_rgb=color, color_name="test",
        pattern=Pattern.SOLID, style=style,
        occasions=[Occasion.CASUAL], seasons=[Season.SUMMER],
        warmth_level=warmth, formality_level=formality,
    )


@pytest.fixture
def items():
    return [
        _make_item("Shirt A", ClothingCategory.SHIRT, (255, 0, 0), Style.CLASSIC),
        _make_item("Shirt B", ClothingCategory.SHIRT, (250, 10, 10), Style.CLASSIC),
        _make_item("Pants A", ClothingCategory.PANTS, (0, 0, 200), Style.MINIMALIST),
        _make_item("Pants B", ClothingCategory.PANTS, (0, 0, 180), Style.MINIMALIST),
        _make_item("Shoes A", ClothingCategory.SHOES, (0, 200, 0), Style.ATHLETIC),
        _make_item("Shoes B", ClothingCategory.SHOES, (0, 180, 0), Style.ATHLETIC),
    ]


def test_fit_assigns_labels(items):
    clusterer = ClothingClusterer(n_clusters=2)
    clusterer.fit(items)
    assert clusterer.labels is not None
    assert len(clusterer.labels) == len(items)


def test_correct_number_of_clusters(items):
    clusterer = ClothingClusterer(n_clusters=3)
    clusterer.fit(items)
    unique_labels = set(clusterer.labels)
    assert len(unique_labels) <= 3


def test_get_cluster_members(items):
    clusterer = ClothingClusterer(n_clusters=2)
    clusterer.fit(items)
    members = clusterer.get_cluster_members(0)
    assert len(members) >= 1
    assert all(isinstance(m, ClothingItem) for m in members)


def test_get_all_clusters(items):
    clusterer = ClothingClusterer(n_clusters=2)
    clusterer.fit(items)
    all_c = clusterer.get_all_clusters()
    total = sum(len(v) for v in all_c.values())
    assert total == len(items)


def test_auto_k_selection(items):
    clusterer = ClothingClusterer(n_clusters=None, max_k=4)
    clusterer.fit(items)
    assert clusterer.n_clusters >= 1


def test_single_item():
    items = [_make_item("Solo", ClothingCategory.SHIRT, (100, 100, 100), Style.CLASSIC)]
    clusterer = ClothingClusterer()
    clusterer.fit(items)
    assert clusterer.n_clusters == 1


def test_get_cluster_raises_on_missing():
    items = [_make_item("X", ClothingCategory.SHIRT, (100, 100, 100), Style.CLASSIC)]
    clusterer = ClothingClusterer()
    clusterer.fit(items)
    with pytest.raises(KeyError):
        clusterer.get_cluster("nonexistent")
