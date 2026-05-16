import os
import streamlit as st
from PIL import Image
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision import predict_and_clean_item
from recommendation_engine.api import RecommendationAPI

st.title("Upload Outfit")

# Setup wardrobe path
wardrobe_path = os.path.join("data", "my_wardrobe.json")
# Ensure directory exists
os.makedirs("data", exist_ok=True)
os.makedirs("outfits", exist_ok=True)

api = RecommendationAPI(wardrobe_path)

def map_prediction_to_item(pred, image):
    # Expect the CNN to return 'baseColour' now
    color_name = pred.get("baseColour", "black").lower()
    rgb = [0, 0, 0] # Default, can be mapped from color_name if needed
    
    cat_str = pred.get("subCategory", "").lower()
    art_str = pred.get("articleType", "").lower()
    
    # Default values
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
    
    # Map Category
    if "innerwear" in cat_str or "loungewear" in cat_str or "bra" in art_str or "brief" in art_str or "underwear" in art_str:
        raise ValueError(f"The model incorrectly classified this as '{art_str}' ({cat_str}). Innerwear is not allowed. Please crop your screenshot closer to the item and try again!")
    elif "top" in cat_str or "shirt" in art_str or "tshirt" in art_str:
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
        
    # Map Season
    season_str = pred.get("season", "").lower()
    if season_str in ["spring", "summer", "autumn", "winter"]:
        item_dict["seasons"] = [season_str]
    elif season_str == "fall":
        item_dict["seasons"] = ["autumn"]
            
    # Map Usage
    usage_str = pred.get("usage", "").lower()
    if usage_str in ["casual", "formal", "sport", "party"]:
        item_dict["occasions"] = [usage_str]
    elif usage_str == "smart casual":
        item_dict["occasions"] = ["casual", "business"]
    elif usage_str == "ethnic":
        item_dict["occasions"] = ["formal"]
        
    return item_dict

uploaded_file = st.file_uploader(
    "Upload clothing image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, width=300)

    if st.button("Predict & Add to Wardrobe"):
        with st.spinner("Analyzing outfit & removing background..."):
            prediction, cleaned_image = predict_and_clean_item(image)
        
        st.success("Prediction Complete!")
        st.image(cleaned_image, caption="Cleaned Image (White Background)", width=300)
        st.json(prediction)
        
        # Save cleaned white-background image
        img_path = os.path.join("outfits", uploaded_file.name)
        cleaned_image.save(img_path)
        
        # Add to Wardrobe
        item_dict = map_prediction_to_item(prediction, cleaned_image)
        item_dict["image_path"] = img_path
        
        try:
            item_id = api.add_item(item_dict)
            st.success(f"Added to Wardrobe successfully! (ID: {item_id})")
        except Exception as e:
            st.error(f"Failed to add to wardrobe: {e}")
else:
    st.info("Please upload an image.")