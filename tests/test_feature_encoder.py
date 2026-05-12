"""Tests for the FeatureEncoder module."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recommendation_engine.data_models import (
    ClothingCategory, Occasion, Season, Style, Pattern, ClothingItem,
)
from recommendation_engine.feature_encoder import FeatureEncoder


@pytest.fixture
def encoder():
    return FeatureEncoder()


@pytest.fixture
def sample_item():
    return ClothingItem(
        name="Test Shirt",
        category=ClothingCategory.SHIRT,
        color_rgb=(128, 64, 200),
        color_name="purple",
        pattern=Pattern.SOLID,
        style=Style.CLASSIC,
        occasions=[Occasion.CASUAL, Occasion.BUSINESS],
        seasons=[Season.SPRING, Season.SUMMER],
        warmth_level=2,
        formality_level=3,
    )


def test_encode_returns_numpy_array(encoder, sample_item):
    vec = encoder.encode(sample_item)
    assert isinstance(vec, np.ndarray)


def test_encode_correct_base_dimensions(encoder, sample_item):
    vec = encoder.encode(sample_item)
    assert len(vec) == encoder.BASE_DIM  # 35


def test_encode_with_cnn_features(encoder):
    item = ClothingItem(
        name="CNN Item",
        category=ClothingCategory.PANTS,
        color_rgb=(100, 100, 100),
        color_name="grey",
        pattern=Pattern.SOLID,
        style=Style.MINIMALIST,
        occasions=[Occasion.CASUAL],
        seasons=[Season.AUTUMN],
        warmth_level=3,
        formality_level=2,
        image_features=[0.1] * 128,
    )
    vec = encoder.encode(item)
    assert len(vec) == encoder.BASE_DIM + 128


def test_encode_values_in_range(encoder, sample_item):
    vec = encoder.encode(sample_item)
    assert vec.min() >= 0.0
    assert vec.max() <= 1.0


def test_encode_many(encoder, sample_item):
    items = [sample_item, sample_item]
    matrix = encoder.encode_many(items)
    assert matrix.shape == (2, encoder.BASE_DIM)


def test_category_one_hot(encoder, sample_item):
    vec = encoder.encode(sample_item)
    cat_slice = vec[:len(encoder.CATEGORIES)]
    assert cat_slice.sum() == 1.0  # exactly one hot


def test_different_items_different_vectors(encoder):
    item1 = ClothingItem(
        name="A", category=ClothingCategory.SHIRT,
        color_rgb=(255, 0, 0), color_name="red",
        pattern=Pattern.STRIPED, style=Style.CLASSIC,
        occasions=[Occasion.CASUAL], seasons=[Season.SUMMER],
        warmth_level=1, formality_level=1,
    )
    item2 = ClothingItem(
        name="B", category=ClothingCategory.SHOES,
        color_rgb=(0, 0, 255), color_name="blue",
        pattern=Pattern.SOLID, style=Style.ATHLETIC,
        occasions=[Occasion.SPORT], seasons=[Season.WINTER],
        warmth_level=5, formality_level=1,
    )
    v1 = encoder.encode(item1)
    v2 = encoder.encode(item2)
    assert not np.allclose(v1, v2)
