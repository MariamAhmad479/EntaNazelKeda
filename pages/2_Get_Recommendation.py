import os
import streamlit as st
import sys
from PIL import Image

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation_engine.api import RecommendationAPI

st.title("✨ Get Recommendation")
st.write("Choose your preferences and generate an outfit recommendation.")

occasion = st.selectbox(
    "Where are you going?",
    ["University", "Work", "Casual outing", "Wedding", "Gym", "Date", "Formal event"]
)

weather = st.selectbox(
    "Weather",
    ["Hot", "Cold", "Rainy", "Mild"]
)

style = st.selectbox(
    "Preferred style",
    ["Casual", "Classic", "Streetwear", "Formal", "Minimal", "Sporty"]
)

gender = st.selectbox(
    "Style category",
    ["Male", "Female", "Unisex"]
)

if st.button("Generate Recommendation"):
    # Initialize API
    wardrobe_path = os.path.join("data", "my_wardrobe.json")
    if not os.path.exists(wardrobe_path):
        st.warning("Your wardrobe is empty! Please upload outfits first.")
        st.stop()
        
    api = RecommendationAPI(wardrobe_path)
    
    # Map inputs to RecommendationEngine enums
    occ_map = {
        "University": "casual",
        "Work": "business",
        "Casual outing": "casual",
        "Wedding": "formal",
        "Gym": "sport",
        "Date": "casual",
        "Formal event": "formal"
    }
    api_occasion = occ_map.get(occasion, "casual")
    
    weather_map = {
        "Hot": {"temperature": 32, "condition": "sunny"},
        "Cold": {"temperature": 5, "condition": "cloudy"},
        "Rainy": {"temperature": 15, "condition": "rainy"},
        "Mild": {"temperature": 22, "condition": "clear"}
    }
    api_weather = weather_map.get(weather, {"temperature": 22, "condition": "clear"})
    
    style_map = {
        "Casual": "minimalist",
        "Classic": "classic",
        "Streetwear": "streetwear",
        "Formal": "classic",
        "Minimal": "minimalist",
        "Sporty": "athletic"
    }
    api_style = style_map.get(style, "classic")
    
    with st.spinner("Styling your outfit..."):
        outfits = api.get_outfits(
            occasion=api_occasion,
            weather=api_weather,
            style=api_style,
            gender=gender,
            top_n=3
        )
        
    if not outfits:
        st.error("Could not find a matching outfit. Try uploading more clothes!")
    else:
        st.success("Recommended Outfits")
        for i, outfit in enumerate(outfits):
            st.subheader(f"Outfit Option {i+1} (Score: {outfit['score']:.2f})")
            
            # Display items in columns
            cols = st.columns(len(outfit["items"]))
            for col, item in zip(cols, outfit["items"]):
                with col:
                    if item.get("image_path") and os.path.exists(item["image_path"]):
                        st.image(Image.open(item["image_path"]), caption=f"{item['name']}", use_container_width=True)
                    else:
                        st.write(f"👕 {item['name']}")
                    st.write(f"Category: {item['category']}")
            
            # Allow saving the outfit
            if st.button(f"Save Outfit {i+1}", key=f"save_{i}"):
                if "saved_outfits" not in st.session_state:
                    st.session_state.saved_outfits = []
                st.session_state.saved_outfits.append(outfit)
                st.success(f"Outfit {i+1} saved to your favorites!")
        
        st.info("These outfits match your selected occasion, weather, and preferred style based on your wardrobe.")