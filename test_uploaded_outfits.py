import os
import sys

# Add the project root to sys.path if not already there
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine.api import RecommendationAPI

def main():
    print("Testing Smart Wardrobe with images in outfits/ folder...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data")
    outfits_dir = os.path.join(project_root, "outfits")
    
    if not os.path.exists(outfits_dir):
        print(f"Error: {outfits_dir} does not exist.")
        sys.exit(1)
        
    # Use a fresh test wardrobe so we don't mess up my_wardrobe.json if we don't want to
    # Or just use an empty memory wardrobe
    test_wardrobe_path = os.path.join(data_dir, "test_outfits_wardrobe.json")
    
    # Remove it if it exists to start fresh
    if os.path.exists(test_wardrobe_path):
        os.remove(test_wardrobe_path)
        
    api = RecommendationAPI(test_wardrobe_path)
    print("Started with empty test wardrobe.")
    
    # List all png/jpg files in outfits/
    images = [f for f in os.listdir(outfits_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print("No images found in the outfits/ folder.")
        sys.exit(0)
        
    print(f"\nFound {len(images)} images to upload:")
    for img in images:
        img_path = os.path.join(outfits_dir, img)
        print(f"\nScanning {img}...")
        try:
            # Provide default metadata since the Vision Model only extracts category and features
            dummy_metadata = {
                "name": f"Uploaded {img}",
                "color_rgb": [0, 0, 0],
                "color_name": "black",
                "pattern": "solid",
                "style": "classic",
                "occasions": ["casual", "formal", "sport", "party"],
                "seasons": ["summer", "winter", "spring", "autumn"],
                "warmth_level": 2,
                "formality_level": 3
            }
            item_id = api.add_item_from_image(img_path, dummy_metadata)
            item = api._wardrobe.get_item(item_id)
            print(f"Extracted: {item.name} ({item.category.value})")
            print(f"   Color: {item.color_name}, Style: {item.style.value}, Formality: {item.formality_level}")
        except Exception as e:
            print(f"Error processing {img}: {e}")
            
    summary = api.get_wardrobe_summary()
    print(f"\nWardrobe populated! Total items: {summary['total_items']}")
    print(f"Categories: {summary['by_category']}")
    
    # Now try to get some recommendations!
    print("\n" + "="*60)
    print("TESTING RECOMMENDATIONS")
    print("="*60)
    
    occasions = ["casual", "formal", "sport", "party"]
    
    for occ in occasions:
        print(f"\nTrying to build a {occ.upper()} outfit...")
        outfits = api.get_outfits(occasion=occ, top_n=3)
        if not outfits:
            print("  -> No suitable outfits found for this occasion.")
        else:
            for i, o in enumerate(outfits, 1):
                print(f"  Outfit #{i} (Score: {o['score']:.2f})")
                for item in o["items"]:
                    print(f"    - {item['name']} ({item['category']})")

if __name__ == "__main__":
    main()
