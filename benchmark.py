import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine.api import RecommendationAPI

def benchmark():
    print("Loading H&M catalog...")
    api = RecommendationAPI("data/hm_catalog.json")
    print(f"Loaded {api.get_wardrobe_summary()['total_items']} items.")
    
    t0 = time.time()
    print("Filtering items for CASUAL and COLD...")
    # Map 'cold' to temp 4
    items = api._wardrobe.get_all_items()
    filtered = api._filter.filter_items(items, occasion="casual", weather={"temperature": 4})
    print(f"Filtered to {len(filtered)} items in {time.time() - t0:.4f} seconds.")
    
    tops = [i for i in filtered if i.category.value in ["shirt", "jacket"]]
    bottoms = [i for i in filtered if i.category.value in ["pants", "skirt", "shorts"]]
    shoes = [i for i in filtered if i.category.value in ["shoes"]]
    print(f"Tops: {len(tops)}, Bottoms: {len(bottoms)}, Shoes: {len(shoes)}")
    
    t0 = time.time()
    print("Generating outfits...")
    outfits = api.get_outfits(occasion="casual", weather={"temperature": 4}, top_n=6)
    print(f"Generated {len(outfits)} outfits in {time.time() - t0:.4f} seconds.")

if __name__ == "__main__":
    benchmark()
