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
