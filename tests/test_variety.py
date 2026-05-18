import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recommendation_engine.data_models import (
    ClothingCategory, Occasion, Season, Style, Pattern, ClothingItem,
)
from recommendation_engine.outfit_generator import OutfitGenerator

def _make(name, cat, rgb=(128, 128, 128)):
    return ClothingItem(
        name=name, category=cat,
        color_rgb=rgb, color_name="test",
        pattern=Pattern.SOLID, style=Style.CLASSIC,
        occasions=[Occasion.CASUAL], seasons=[Season.SUMMER],
        warmth_level=2, formality_level=3,
    )

def test_strict_dress_variety_rule():
    # Large wardrobe containing 3 dresses, 4 shirts, 3 pants, 2 skirts, 2 shorts, and shoes
    items = [
        # Shoes
        _make("Black Shoes", ClothingCategory.SHOES),
        _make("White Sneakers", ClothingCategory.SHOES),
        
        # Dresses
        _make("Summer Dress", ClothingCategory.DRESS),
        _make("Floral Dress", ClothingCategory.DRESS),
        _make("Party Dress", ClothingCategory.DRESS),
        
        # Shirts
        _make("Cotton Shirt", ClothingCategory.SHIRT),
        _make("Linen Shirt", ClothingCategory.SHIRT),
        _make("Silk Blouse", ClothingCategory.SHIRT),
        _make("Polo Shirt", ClothingCategory.SHIRT),
        
        # Pants (trousers)
        _make("Blue Jeans", ClothingCategory.PANTS),
        _make("Khaki Chinos", ClothingCategory.PANTS),
        _make("Black Trousers", ClothingCategory.PANTS),
        
        # Skirts
        _make("Pleated Skirt", ClothingCategory.SKIRT),
        _make("Denim Skirt", ClothingCategory.SKIRT),
        
        # Shorts
        _make("Cargo Shorts", ClothingCategory.SHORTS),
        _make("Gym Shorts", ClothingCategory.SHORTS),
    ]
    
    gen = OutfitGenerator(min_score=0.0)
    
    # Generate 5 outfits
    outfits = gen.generate(items, top_n=5)
    
    dress_outfits_count = 0
    bottom_categories_seen = []
    
    for o in outfits:
        cats = [item.category for item in o.items]
        is_dress = ClothingCategory.DRESS in cats
        if is_dress:
            dress_outfits_count += 1
            # Assure no dress outfit has bottoms
            assert not any(c in cats for c in [ClothingCategory.PANTS, ClothingCategory.SKIRT, ClothingCategory.SHORTS])
        else:
            # Standard outfit: check bottom category
            bottom_item = next((item for item in o.items if item.category in [ClothingCategory.PANTS, ClothingCategory.SKIRT, ClothingCategory.SHORTS]), None)
            if bottom_item:
                bottom_categories_seen.append(bottom_item.category)
                
    # 1. Enforce variety: AT MOST 1 DRESS allowed!
    assert dress_outfits_count <= 1
    
    # 2. Check bottoms variety
    first_three_bottom_cats = bottom_categories_seen[:3]
    unique_first_three = set(first_three_bottom_cats)
    
    # The first 3 standard outfits should alternate bottom categories and thus contain at least 2 distinct types of bottoms
    assert len(unique_first_three) >= 2


def test_no_more_than_two_repeated_items():
    # Create a wardrobe where combinations could naturally share items
    # E.g. Shirt 1, Shirt 2, Shirt 3
    # Pants 1, Pants 2
    # Shoes 1
    # Accessory 1, Accessory 2
    # Jacket 1
    items = [
        _make("Shirt 1", ClothingCategory.SHIRT),
        _make("Shirt 2", ClothingCategory.SHIRT),
        _make("Shirt 3", ClothingCategory.SHIRT),
        _make("Pants 1", ClothingCategory.PANTS),
        _make("Pants 2", ClothingCategory.PANTS),
        _make("Shoes 1", ClothingCategory.SHOES),
        _make("Accessory 1", ClothingCategory.ACCESSORY),
        _make("Accessory 2", ClothingCategory.ACCESSORY),
        _make("Jacket 1", ClothingCategory.JACKET),
    ]
    
    gen = OutfitGenerator(min_score=0.0)
    outfits = gen.generate(items, top_n=5)
    
    # Assert that no two outfits in the recommended set share more than 2 items
    for i in range(len(outfits)):
        for j in range(i + 1, len(outfits)):
            o1 = outfits[i]
            o2 = outfits[j]
            o1_ids = {item.id for item in o1.items}
            o2_ids = {item.id for item in o2.items}
            overlap = len(o1_ids.intersection(o2_ids))
            assert overlap <= 2, f"Outfits share {overlap} items (>2): {o1} and {o2}"


def test_style_randomized_selection(tmp_path):
    import json
    from recommendation_engine.api import RecommendationAPI
    
    # Create 15 classic shirts, 15 classic pants, 1 shoes
    # This will generate many distinct outfits with Style.CLASSIC
    items = []
    for i in range(15):
        items.append(_make(f"Classic Shirt {i}", ClothingCategory.SHIRT))
        items.append(_make(f"Classic Pants {i}", ClothingCategory.PANTS))
    items.append(_make("Shoes 1", ClothingCategory.SHOES))
    
    # Save to a temporary JSON file
    wardrobe_data = [item.to_dict() for item in items]
    wardrobe_file = tmp_path / "mock_wardrobe.json"
    with open(wardrobe_file, "w") as f:
        json.dump(wardrobe_data, f)
        
    api = RecommendationAPI(str(wardrobe_file))
    
    # Call get_outfits with style="classic" multiple times
    # It should return 3 outfits each time, and across multiple runs it should return different subsets
    all_runs_outfits = []
    for _ in range(10):
        outfits = api.get_outfits(style="classic")
        assert len(outfits) <= 3
        # Extract summaries of the recommended outfits
        summaries = sorted([o["summary"] for o in outfits])
        all_runs_outfits.append(tuple(summaries))
        
    # Check that we got at least some randomization (not all runs returned identical outfits)
    unique_runs = set(all_runs_outfits)
    assert len(unique_runs) > 1, "Expected different randomized subsets when querying the same style."
