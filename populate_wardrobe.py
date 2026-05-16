import os
import sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision import predict_item
from recommendation_engine.api import RecommendationAPI

def map_color_to_rgb(color_name):
    color_map = {
        "black": [0, 0, 0],
        "white": [255, 255, 255],
        "gray": [128, 128, 128],
        "grey": [128, 128, 128],
        "silver": [192, 192, 192],
        "red": [255, 0, 0],
        "maroon": [128, 0, 0],
        "blue": [0, 0, 255],
        "navy blue": [0, 0, 128],
        "navy": [0, 0, 128],
        "light blue": [173, 216, 230],
        "green": [0, 128, 0],
        "olive": [128, 128, 0],
        "yellow": [255, 255, 0],
        "orange": [255, 165, 0],
        "pink": [255, 192, 203],
        "purple": [128, 0, 128],
        "brown": [165, 42, 42],
        "beige": [245, 245, 220],
        "gold": [255, 215, 0]
    }
    return color_map.get(color_name.lower(), [128, 128, 128])

def map_prediction_to_item(pred, filename, image):
    # Expect the CNN to return 'baseColour' now
    color_name = pred.get("baseColour", "black").lower()
    rgb = map_color_to_rgb(color_name)
    
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
    
    if any(word in art_str for word in ["dress", "saree", "jumpsuit", "romper", "gown"]):
        item_dict["category"] = "dress"
    elif any(word in art_str for word in ["shirt", "tshirt", "top", "blouse", "kurta", "tunic", "tank"]):
        item_dict["category"] = "shirt"
    elif any(word in art_str for word in ["pant", "jeans", "trouser", "track pant", "legging", "jogger"]):
        item_dict["category"] = "pants"
    elif "short" in art_str:
        item_dict["category"] = "shorts"
    elif "skirt" in art_str:
        item_dict["category"] = "skirt"
    elif any(word in art_str for word in ["shoe", "sneaker", "boot", "sandal", "heel", "flip flop", "flat"]):
        item_dict["category"] = "shoes"
    elif any(word in art_str for word in ["jacket", "coat", "sweater", "sweatshirt", "hoodie", "blazer", "cardigan"]):
        item_dict["category"] = "jacket"
    elif any(word in art_str for word in ["sock", "tie", "belt", "hat", "cap", "bag", "jewel", "watch", "accessory", "bra", "briefs", "dupatta", "scarf"]):
        item_dict["category"] = "accessory"
    elif "top" in cat_str:
        item_dict["category"] = "shirt"
    elif "bottom" in cat_str:
        item_dict["category"] = "pants"
    elif "shoe" in cat_str or "foot" in cat_str or "flip flops" in cat_str:
        item_dict["category"] = "shoes"
    elif "dress" in cat_str:
        item_dict["category"] = "dress"
    elif "outer" in cat_str:
        item_dict["category"] = "jacket"
    elif "innerwear" in cat_str:
        item_dict["category"] = "accessory"
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
            if prediction.get("subCategory", "").lower() == "innerwear":
                print(f"Skipping innerwear: {filename}")
                continue
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
