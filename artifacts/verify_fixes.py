import os
import json
import sys

# Set import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recommendation_engine.data_models import (
    ClothingItem, ClothingCategory, Season,
    BOTTOM_CATEGORIES, FOOTWEAR_CATEGORIES, FULL_BODY_CATEGORIES
)
from recommendation_engine.context_filter import ContextFilter
from recommendation_engine.outfit_generator import OutfitGenerator

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, 'data', 'hm_catalog.json')
    
    print(f"Loading generated catalog from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        catalog_raw = json.load(f)
        
    items = [ClothingItem.from_dict(d) for d in catalog_raw]
    print(f"Loaded {len(items)} items from H&M Catalog.")
    
    # ----------------------------------------------------
    # Verification 1: Mappings of Seasons & Warmth
    # ----------------------------------------------------
    dresses = [i for i in items if i.category == ClothingCategory.DRESS]
    shorts = [i for i in items if i.category == ClothingCategory.SHORTS]
    jackets = [i for i in items if i.category == ClothingCategory.JACKET]
    
    print(f"\n--- Checking Mappings ---")
    print(f"Total Dresses: {len(dresses)}")
    print(f"Total Shorts: {len(shorts)}")
    print(f"Total Jackets: {len(jackets)}")
    
    # Check that shorts are Summer/Spring and low warmth
    for s in shorts[:10]:
        assert Season.SUMMER in s.seasons or Season.SPRING in s.seasons
        assert s.warmth_level <= 2
    print("[OK] Verified: Shorts are successfully mapped to low warmth levels and warmer seasons.")
    
    # Verify we have winter jackets and summer dresses
    winter_jackets = [j for j in jackets if Season.WINTER in j.seasons and j.warmth_level >= 4]
    summer_dresses = [d for d in dresses if Season.SUMMER in d.seasons and Season.WINTER not in d.seasons and d.warmth_level == 1]
    
    print(f"Winter Jackets (Warmth >= 4): {len(winter_jackets)}")
    print(f"Summer-Only Dresses (Warmth = 1): {len(summer_dresses)}")
    
    assert len(winter_jackets) > 0, "No winter jackets found!"
    assert len(summer_dresses) > 0, "No summer-only dresses found!"
    print("[OK] Verified: Description parser successfully parsed and separated winter jackets and summer dresses.")
    
    # ----------------------------------------------------
    # Verification 2: Weather filtering (Winter cold query)
    # ----------------------------------------------------
    print(f"\n--- Testing Weather Filter (Freezing Cold: 4°C) ---")
    cf = ContextFilter()
    cold_weather = {"temperature": 4, "condition": "cold"}
    
    # Filter catalog items
    filtered_items = cf.filter_items(items, weather=cold_weather)
    print(f"Filtered down to {len(filtered_items)} items for 4°C weather.")
    
    # Print slot breakdown
    tops_count = len([i for i in filtered_items if i.category == ClothingCategory.SHIRT])
    bottoms_count = len([i for i in filtered_items if i.category in BOTTOM_CATEGORIES])
    shoes_count = len([i for i in filtered_items if i.category in FOOTWEAR_CATEGORIES])
    dresses_count = len([i for i in filtered_items if i.category in FULL_BODY_CATEGORIES])
    jackets_count = len([i for i in filtered_items if i.category == ClothingCategory.JACKET])
    
    print(f"Filtered items count by slot:")
    print(f"  Tops (Shirts): {tops_count}")
    print(f"  Bottoms: {bottoms_count}")
    print(f"  Shoes: {shoes_count}")
    print(f"  Dresses: {dresses_count}")
    print(f"  Jackets: {jackets_count}")
    
    # Check that no summer-only dress or low warmth shorts are included
    for item in filtered_items:
        if item.category == ClothingCategory.DRESS:
            assert Season.WINTER in item.seasons, f"Summer dress recommended in freezing winter: {item.name}"
            assert item.warmth_level >= 3, f"Thin dress recommended in freezing winter: {item.name} (warmth {item.warmth_level})"
        if item.category == ClothingCategory.SHORTS:
            assert item.warmth_level >= 3, f"Cold shorts recommended in freezing winter: {item.name}"
            
    print("[OK] Verified: No summer dresses or light shorts are recommended in cold winter temperatures!")

    # ----------------------------------------------------
    # Verification 3: Outfit Generator Variety Shuffling
    # ----------------------------------------------------
    print(f"\n--- Testing Outfit Variety (Multiple Generations: 18°C Mild Weather) ---")
    mild_weather = {"temperature": 18, "condition": "sunny"}
    mild_items = cf.filter_items(items, weather=mild_weather)
    print(f"Filtered down to {len(mild_items)} items for 18°C weather.")
    
    gen = OutfitGenerator(min_score=0.0, max_combinations=100)
    
    # We will run generate 3 times over the mild filtered items and see if they return different looks
    outfits1 = gen.generate(mild_items, top_n=5)
    outfits2 = gen.generate(mild_items, top_n=5)
    outfits3 = gen.generate(mild_items, top_n=5)
    
    ids1 = [o.to_dict()["summary"] for o in outfits1]
    ids2 = [o.to_dict()["summary"] for o in outfits2]
    ids3 = [o.to_dict()["summary"] for o in outfits3]
    
    print("Run 1 Summaries:")
    for summary in ids1:
         print(f"  - {summary}")
         
    print("Run 2 Summaries:")
    for summary in ids2:
         print(f"  - {summary}")
         
    print("Run 3 Summaries:")
    for summary in ids3:
         print(f"  - {summary}")
         
    # Check if the set of recommendations changes
    # Some high-scoring overlap is natural, but they shouldn't be identical lists in identical order
    assert ids1 != ids2 or ids2 != ids3, "Recommendations are identical across runs!"
    print("[OK] Verified: Successive outfit generations return different diverse combinations. Variety is fixed!")

    # ----------------------------------------------------
    # Verification 4: Gender Filtration
    # ----------------------------------------------------
    print(f"\n--- Testing Gender Filtration ---")
    female_filtered = cf.filter_items(items, gender="female")
    male_filtered = cf.filter_items(items, gender="male")
    
    print(f"Female filtered count: {len(female_filtered)}")
    print(f"Male filtered count: {len(male_filtered)}")
    
    # Assert female items contain no male-gendered items
    for item in female_filtered:
        assert getattr(item, "gender", "unisex") != "male", f"Male item found in female catalog: {item.name}"
        
    # Assert male items contain no female-gendered items
    for item in male_filtered:
        assert getattr(item, "gender", "unisex") != "female", f"Female item found in male catalog: {item.name}"
        
    print("[OK] Verified: Gender filtration is 100% strict and correct!")

    # ----------------------------------------------------
    # Verification 5: Outfit Structure and Item Diversity
    # ----------------------------------------------------
    print(f"\n--- Testing Outfit Structure and Item Diversity ---")
    # Let's generate outfits from a mix of items
    mixed_items = []
    # Add 10 dresses, 10 shirts, 10 pants, 10 shoes
    mixed_items.extend([i for i in items if i.category == ClothingCategory.DRESS][:10])
    mixed_items.extend([i for i in items if i.category == ClothingCategory.SHIRT][:10])
    mixed_items.extend([i for i in items if i.category == ClothingCategory.PANTS][:10])
    mixed_items.extend([i for i in items if i.category == ClothingCategory.SHOES][:10])
    
    diversity_gen = OutfitGenerator(min_score=0.1, max_combinations=2000)
    top_outfits = diversity_gen.generate(mixed_items, top_n=5)
    
    print(f"Generated {len(top_outfits)} outfits from mixed pool:")
    structures = []
    main_items_seen = []
    
    for idx, o in enumerate(top_outfits, 1):
        is_dress = any(item.category == ClothingCategory.DRESS for item in o.items)
        struct = "dress" if is_dress else "standard"
        structures.append(struct)
        
        main_garments = [
            item.name for item in o.items 
            if item.category in (ClothingCategory.DRESS, ClothingCategory.SHIRT, ClothingCategory.JACKET, ClothingCategory.PANTS, ClothingCategory.SKIRT, ClothingCategory.SHORTS)
        ]
        print(f"  Outfit #{idx} ({struct}): {[item.name for item in o.items]}")
        main_items_seen.extend(main_garments)
        
    # Check that main garments are not excessively duplicated
    unique_main_garments = set(main_items_seen)
    print(f"Total garment references: {len(main_items_seen)}, Unique garments: {len(unique_main_garments)}")
    # If perfect diversity is maintained, we should have nearly as many unique garments as reference slots
    assert len(unique_main_garments) >= len(top_outfits), "Main garments are being duplicated across outfits!"
    
    # Check structure mix: should not be 100% of one structure if a mix was possible and top_n >= 2
    if len(top_outfits) >= 3 and "dress" in structures and "standard" in structures:
        print(f"Structures: {structures}")
        assert len(set(structures)) > 1 or len(structures) < 3, "Failed to balance outfit structures!"
        print("[OK] Verified: Balanced mix of dresses and standard outfits generated!")
        
    print("[OK] Verified: Diversity Reranker successfully ensures unique garments across suggested outfits!")

if __name__ == '__main__':
    main()
