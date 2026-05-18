import pytest
from recommendation_engine.context_filter import ContextFilter
from recommendation_engine.data_models import ClothingItem, ClothingCategory, Style, Pattern, Season, Occasion

@pytest.fixture
def sample_wardrobe():
    return [
        ClothingItem(
            name="Light Tank Top",
            category=ClothingCategory.SHIRT,
            color_rgb=(255, 255, 255),
            color_name="white",
            pattern=Pattern.SOLID,
            style=Style.MINIMALIST,
            occasions=[Occasion.CASUAL],
            seasons=[Season.SUMMER],
            warmth_level=1,
            formality_level=1
        ),
        ClothingItem(
            name="Heavy Winter Jacket",
            category=ClothingCategory.JACKET,
            color_rgb=(0, 0, 128),
            color_name="navy",
            pattern=Pattern.SOLID,
            style=Style.CLASSIC,
            occasions=[Occasion.OUTDOOR],
            seasons=[Season.WINTER],
            warmth_level=5,
            formality_level=3
        ),
        ClothingItem(
            name="Wool Sweatshirt",
            category=ClothingCategory.SHIRT,
            color_rgb=(100, 100, 100),
            color_name="grey",
            pattern=Pattern.SOLID,
            style=Style.STREETWEAR,
            occasions=[Occasion.CASUAL],
            seasons=[Season.WINTER, Season.AUTUMN],
            warmth_level=4,
            formality_level=2
        ),
        ClothingItem(
            name="Summer Floral Dress",
            category=ClothingCategory.DRESS,
            color_rgb=(255, 192, 203),
            color_name="pink",
            pattern=Pattern.FLORAL,
            style=Style.BOHEMIAN,
            occasions=[Occasion.PARTY],
            seasons=[Season.SUMMER],
            warmth_level=1,
            formality_level=3
        ),
    ]

def test_hot_weather_excludes_jackets_and_sweaters(sample_wardrobe):
    cf = ContextFilter()
    
    # Test at 30 degrees (hot weather)
    weather = {"temperature": 30}
    filtered = cf.filter_items(sample_wardrobe, weather=weather)
    
    # Assertions
    item_names = [i.name.lower() for i in filtered]
    
    # Should contain light summer stuff
    assert "light tank top" in item_names
    assert "summer floral dress" in item_names
    
    # Should NOT contain jackets or wool/sweatshirt items
    assert "heavy winter jacket" not in item_names
    assert "wool sweatshirt" not in item_names
    
    # Verify no JACKET category exists in the result
    categories = {i.category for i in filtered}
    assert ClothingCategory.JACKET not in categories

def test_threshold_at_24_degrees(sample_wardrobe):
    cf = ContextFilter()
    
    # Test at 24 degrees (now considered hot in our new threshold of 23)
    weather = {"temperature": 24}
    filtered = cf.filter_items(sample_wardrobe, weather=weather)
    
    categories = {i.category for i in filtered}
    assert ClothingCategory.JACKET not in categories
    
    item_names = [i.name.lower() for i in filtered]
    assert "wool sweatshirt" not in item_names

def test_mild_weather_allows_cardigans_if_warmth_matches(sample_wardrobe):
    cf = ContextFilter()
    
    # Test at 15 degrees (mild/cool weather)
    # Warmth range for 15 is (3, 4)
    weather = {"temperature": 15}
    filtered = cf.filter_items(sample_wardrobe, weather=weather)
    
    item_names = [i.name.lower() for i in filtered]
    assert "wool sweatshirt" in item_names
