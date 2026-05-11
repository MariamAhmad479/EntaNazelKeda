import os
import sys

# Add the project root to sys.path if not already there
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine.api import RecommendationAPI
from nlp.inference import NLPInference

def main():
    print("Initializing Smart Wardrobe Recommendation System...")
    
    # 1. Initialize Person 2's Recommendation API
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    wardrobe_file = os.path.join(data_dir, "sample_wardrobe.json")
    
    if not os.path.exists(wardrobe_file):
        print(f"Error: Wardrobe file not found at {wardrobe_file}")
        sys.exit(1)
        
    api = RecommendationAPI(wardrobe_file)
    print(f"Loaded Recommendation API with {api.get_wardrobe_summary()['total_items']} items.")
    
    # 2. Initialize Person 3's NLP Model
    model_dir = os.path.join(os.path.dirname(__file__), "nlp", "saved_models")
    if not os.path.exists(os.path.join(model_dir, "nlp_model.pth")):
        print("Error: NLP model weights not found. Please run 'python nlp/train.py' first.")
        sys.exit(1)
        
    nlp_model = NLPInference(model_dir=model_dir)
    print("Loaded NLP Inference Model.\n")
    
    print("="*60)
    print("SMART WARDROBE AI ASSISTANT")
    print("Type your query (or 'quit' to exit)")
    print("Example: 'I need an outfit for a formal event, it's very cold outside.'")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.strip().lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_input.strip():
            continue
            
        # Extract intent using NLP
        print("\n🤖 Thinking...")
        extracted_params = nlp_model.predict(user_input)
        
        occasion = extracted_params["occasion"]
        weather = extracted_params["weather"]
        
        print(f"[Extracted] Occasion: {occasion.upper()}, Weather: {extracted_params['weather_class'].upper()} ({weather['temperature']}°C)")
        
        # Pass to Recommendation Engine
        outfits = api.get_outfits(occasion=occasion, weather=weather, top_n=3)
        
        if not outfits:
            print("No suitable outfits found in your wardrobe for this request.")
            continue
            
        print("\n👗 Here are your top 3 outfit recommendations:")
        for i, outfit in enumerate(outfits, 1):
            print(f"\nOutfit #{i} (Score: {outfit['score']:.2f})")
            print(f"  Summary: {outfit['summary']}")
            for item in outfit["items"]:
                print(f"    - {item['name']} ({item['color_name']} {item['category']})")

if __name__ == "__main__":
    main()
