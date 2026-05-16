import os
import streamlit as st
import sys
from PIL import Image

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation_engine.api import RecommendationAPI
from recommendation_engine.location_weather import load_locations, get_location_details, map_category_to_occasion, fetch_realtime_weather

st.title("✨ Get Recommendation")
st.write("Choose your preferences and generate an outfit recommendation.")

mode = st.radio("How would you like to get recommendations?", ["Location-Based (Auto Weather)", "Manual Input"])

if mode == "Manual Input":
    occasion = st.selectbox(
        "Where are you going?",
        ["University", "Work", "Casual outing", "Wedding", "Gym", "Date", "Formal event"]
    )
    
    weather = st.selectbox(
        "Weather",
        ["Hot", "Cold", "Rainy", "Mild"]
    )
else:
    csv_path = os.path.join("data", "egypt_places_dummy.csv")
    df_locations = load_locations(csv_path)
    if not df_locations.empty:
        places = df_locations['place_name'].tolist()
        selected_place = st.selectbox("Select your destination:", places)
    else:
        st.error("Could not load locations data.")
        selected_place = None

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
    if mode == "Manual Input":
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
    else:
        if selected_place:
            with st.spinner("Fetching real-time weather & occasion info..."):
                loc_details = get_location_details(df_locations, selected_place)
                cat = loc_details.get("category", "")
                lat = loc_details.get("lat", 30.0444)
                lng = loc_details.get("lng", 31.2357)
                
                api_occasion = map_category_to_occasion(cat)
                api_weather = fetch_realtime_weather(lat, lng)
                
                st.info(f"📍 **Location:** {selected_place} ({cat.replace('_', ' ').title()})  \n"
                        f"👗 **Mapped Occasion:** {api_occasion.title()}  \n"
                        f"🌤️ **Current Weather:** {api_weather['temperature']}°C, {api_weather['condition'].title()}")
        else:
            st.error("Please select a valid location.")
            st.stop()
    
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