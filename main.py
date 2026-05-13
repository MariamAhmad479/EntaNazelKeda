import os
import sys

# Add the project root to sys.path if not already there
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine.api import RecommendationAPI

def main():
    print("Initializing Smart Wardrobe Recommendation System...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data")
    
    # ------------------------------------------------------------------
    # 1. Initialize Recommendation API
    #    We start with a realistic, empty (or previously saved) personal wardrobe.
    # ------------------------------------------------------------------
    my_wardrobe = os.path.join(data_dir, "my_wardrobe.json")
    
    print("[Wardrobe] Loading personal wardrobe...")
    api = RecommendationAPI(my_wardrobe)
    print(f"[Wardrobe] Loaded {api.get_wardrobe_summary()['total_items']} items.")
    
    # Print category breakdown
    summary = api.get_wardrobe_summary()
    print(f"\nWardrobe categories: {summary['by_category']}")
    
    # ------------------------------------------------------------------
    # 2. Interactive loop
    # ------------------------------------------------------------------
    print("="*60)
    print("SMART WARDROBE AI ASSISTANT")
    print("Commands:")
    print("  - Type 'upload <image_path>' to add a new clothing item to your wardrobe")
    print("  - Type '1' for a Casual outfit")
    print("  - Type '2' for a Formal outfit")
    print("  - Type '3' for a Sport outfit")
    print("  - Type '4' for a Party outfit")
    print("  - Type 'quit' to exit")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_input:
            continue
            
        # Handle Upload
        if user_input.lower().startswith("upload "):
            img_path = user_input[7:].strip()
            if not os.path.exists(img_path):
                print(f"Error: Image '{img_path}' not found.")
                continue
            
            print(f"\n📸 Scanning {img_path} with Vision Model...")
            try:
                item_id = api.add_item_from_image(img_path)
                item = api._wardrobe.get_item(item_id)
                print(f"✅ Added new item to wardrobe: {item.name} ({item.category.value})")
                print(f"Wardrobe now has {api._wardrobe.size} items.")
            except Exception as e:
                print(f"Error processing image: {e}")
            continue

        # Handle Recommendations (Menu)
        occasion = None
        if user_input == '1':
            occasion = "casual"
        elif user_input == '2':
            occasion = "formal"
        elif user_input == '3':
            occasion = "sport"
        elif user_input == '4':
            occasion = "party"
        else:
            print("Invalid command. Please type 'upload <path>' or choose 1, 2, 3, or 4.")
            continue
            
        print(f"\n🤖 Building a {occasion.upper()} outfit from your wardrobe...")
        
        # Pass to Recommendation Engine
        outfits = api.get_outfits(occasion=occasion, top_n=3)
        
        if not outfits:
            print("No suitable outfits found in your wardrobe for this request.")
            print("(Remember: You need to upload at least a top + bottom + shoes to get an outfit!)")
            continue
            
        print("\n👗 Here are your top 3 outfit recommendations:")
        for i, outfit in enumerate(outfits, 1):
            print(f"\nOutfit #{i} (Score: {outfit['score']:.2f})")
            print(f"  Summary: {outfit['summary']}")
            for item in outfit["items"]:
                print(f"    - {item['name']} ({item['color_name']} {item['category']})")

if __name__ == "__main__":
    main()
