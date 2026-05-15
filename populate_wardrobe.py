import os
import sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision import predict_item
from recommendation_engine.api import RecommendationAPI

def map_prediction_to_item(pred, filename, image):
    # Expect the CNN to return 'baseColour' now
    color_name = pred.get("baseColour", "black").lower()
    rgb = [0, 0, 0] # Default, can be mapped from color_name if needed
    
    cat_str = pred.get("subCategory", "").lower()
    art_str = pred.get("articleType", "").lower()
    
    item_dict = {
        "name": f"{color_name.capitalize()} {art_str.capitalize()}",
        "color_rgb": rgb,
        "color_name": color_name,
        "pattern": "solid",
        "style": "classic",
        "occasions": ["casual"],
        "seasons": ["summer"],
        "warmth_level": 2,
        "formality_level": 3
    }
    
    if "top" in cat_str or "shirt" in art_str or "tshirt" in art_str:
        item_dict["category"] = "shirt"
    elif "bottom" in cat_str or "pant" in art_str or "jeans" in art_str:
        item_dict["category"] = "pants"
    elif "dress" in cat_str or "dress" in art_str:
        item_dict["category"] = "dress"
    elif "shoe" in cat_str or "foot" in cat_str or "sneaker" in art_str:
        item_dict["category"] = "shoes"
    elif "jacket" in art_str or "coat" in art_str or "outer" in cat_str:
        item_dict["category"] = "jacket"
    elif "skirt" in art_str:
        item_dict["category"] = "skirt"
    elif "short" in art_str:
        item_dict["category"] = "shorts"
    else:
        item_dict["category"] = "accessory"
        
    season_str = pred.get("season", "").lower()
    if season_str in ["spring", "summer", "autumn", "winter"]:
        item_dict["seasons"] = [season_str]
    elif season_str == "fall":
        item_dict["seasons"] = ["autumn"]
            
    usage_str = pred.get("usage", "").lower()
    if usage_str in ["casual", "formal", "sport", "party"]:
        item_dict["occasions"] = [usage_str]
    elif usage_str == "smart casual":
        item_dict["occasions"] = ["casual", "business"]
    elif usage_str == "ethnic":
        item_dict["occasions"] = ["formal"]
        
    return item_dict

def main():
    wardrobe_path = os.path.join("data", "my_wardrobe.json")
    os.makedirs("data", exist_ok=True)
    
    if os.path.exists(wardrobe_path):
        os.remove(wardrobe_path)
        
    api = RecommendationAPI(wardrobe_path)
    outfits_dir = "outfits"
    
    count = 0
    for filename in os.listdir(outfits_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(outfits_dir, filename)
        try:
            image = Image.open(img_path)
            prediction = predict_item(image)
            item_dict = map_prediction_to_item(prediction, filename, image)
            item_dict["image_path"] = img_path
            
            api.add_item(item_dict)
            print(f"Added {filename}: {item_dict['name']} ({prediction})")
            count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    print(f"Successfully added {count} items to the wardrobe.")

if __name__ == "__main__":
    main()
