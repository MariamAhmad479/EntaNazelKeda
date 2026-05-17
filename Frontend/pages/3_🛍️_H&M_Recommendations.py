import streamlit as st
import sys
import os

# Ensure the parent directory is in the path to import utils
FRONTEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(FRONTEND_DIR)
sys.path.append(FRONTEND_DIR)
sys.path.append(PROJECT_ROOT)

from utils import (
    load_css,
    render_sidebar,
    get_user_api,
    save_outfit,
    DATA_DIR
)

st.set_page_config(page_title="H&M Recommendations", page_icon="🛍️", layout="wide")
load_css()
render_sidebar()

st.title("🛍️ H&M Catalog Recommendations")
st.write("Browse H&M's global catalog and generate immediate style looks instantly! Simply specify the occasion, weather, and style tags to get compatible outfit ideas, then save the ones you love directly to your closet.")

# Category emojis for outfit placeholders
CATEGORY_EMOJIS = {
    "shirt": "👕",
    "pants": "👖",
    "dress": "👗",
    "shoes": "👟",
    "jacket": "🧥",
    "skirt": "👗",
    "accessory": "👜",
    "shorts": "🩳"
}

# ── Cache H&M Catalog API ─────────────────────────────────────────────────────
@st.cache_resource
def get_hm_api():
    """Load and cache the H&M global store catalog for high performance."""
    hm_catalog_path = os.path.join(DATA_DIR, "hm_catalog.json")
    feedback_path = os.path.join(DATA_DIR, "hm_feedback_log.json")
    
    from recommendation_engine.api import RecommendationAPI
    return RecommendationAPI(wardrobe_path=hm_catalog_path, feedback_path=feedback_path)

hm_api = get_hm_api()
username = st.session_state.get("username") or "guest"

# ── Quick Style Generator Layout ──────────────────────────────────────────────
col_form, col_results = st.columns([1, 1.8], gap="large")

with col_form:
    st.markdown("""
    <div style="background-color: #f7edd8; border: 1.5px solid rgba(212,160,23,0.4); border-radius: 20px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.04);">
        <h3 style="color: #2b1b11; margin: 0 0 15px 0; font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 700;">
            ✨ Quick Style Generator
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Form fields
    occasions = ["Casual", "Formal", "Business", "Sport", "Party", "Outdoor"]
    selected_occasion = st.selectbox("What is the occasion?", occasions)
    
    # Weather Seasons Mapping
    WEATHER_SEASONS = {
        "☀️ Summer (Hot, ≥ 25°C)": 30,
        "🌸 Spring (Mild, 16–24°C)": 20,
        "🍁 Autumn (Chilly, 6–15°C)": 12,
        "❄️ Winter (Freezing, ≤ 5°C)": 3
    }
    
    # Initialize session state for selected weather index
    if "selected_season" not in st.session_state:
        st.session_state["selected_season"] = "🌸 Spring (Mild, 16–24°C)"
        
    col_season, col_rand = st.columns([2.5, 1.5])
    with col_season:
        selected_season = st.selectbox(
            "Select Season Context:",
            list(WEATHER_SEASONS.keys()),
            index=list(WEATHER_SEASONS.keys()).index(st.session_state["selected_season"])
        )
        st.session_state["selected_season"] = selected_season
    with col_rand:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🎲 Randomize", help="Select a random season context!", key="randomize_weather_btn"):
            import random
            random_season = random.choice(list(WEATHER_SEASONS.keys()))
            st.session_state["selected_season"] = random_season
            st.rerun()
            
    selected_temp = WEATHER_SEASONS[selected_season]
    
    styles = ["All", "Classic", "Streetwear", "Minimalist", "Preppy", "Athletic", "Bohemian"]
    selected_style = st.selectbox("Preferred Style Tag:", styles)
    
    genders = ["Unisex", "Male", "Female"]
    selected_gender = st.selectbox("Garment Cut / Gender Preference:", genders)
    
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("Generate H&M Looks 🚀")

with col_results:
    st.subheader("Outfit Recommendations")
    
    if generate_btn:
        occasion_val = selected_occasion.lower()
        style_val = None if selected_style == "All" else selected_style.lower()
        gender_val = selected_gender
        
        weather_dict = {
            "temperature": selected_temp,
            "condition": "clear"
        }
        
        # Log previous unseen/unsaved outfits as 'reject'
        if "hm_outfits" in st.session_state:
            saved_indices = st.session_state.get("saved_hm_indices", set())
            for idx, outfit in enumerate(st.session_state["hm_outfits"]):
                if idx not in saved_indices:
                    try:
                        user_api = get_user_api(username)
                        user_api.submit_feedback(outfit["outfit_id"], "reject")
                    except Exception as e:
                        print(f"[Feedback] Failed to log reject on skip: {e}")
        # Clear the saved indices set for the new batch
        st.session_state["saved_hm_indices"] = set()
        
        with st.spinner("AI is searching the H&M database for compatible outfits..."):
            try:
                # Retrieve matching looks directly from global catalog
                outfits = hm_api.get_outfits(
                    occasion=occasion_val,
                    weather=weather_dict,
                    style=style_val,
                    gender=gender_val,
                    top_n=5
                )
                
                if not outfits:
                    st.info(f"No fully compatible outfits matching '{selected_occasion}' outfits for {selected_temp}°C weather were found in H&M's catalog. Try relaxing the temperature or style filter!")
                else:
                    st.session_state["hm_outfits"] = outfits
                    st.session_state["hm_context"] = f"H&M {selected_occasion} at {selected_temp}°C ({selected_style} style)"
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
                
    # Display the outfits if they exist in state
    if "hm_outfits" in st.session_state:
        outfits = st.session_state["hm_outfits"]
        context_prompt = st.session_state.get("hm_context", "H&M Generator")
        
        for idx, outfit in enumerate(outfits):
            # Score and details banner
            st.markdown(f"""
            <div style="background-color: #fcf9f2; border: 2px solid #d4a017; border-radius: 16px 16px 0 0; padding: 12px 16px; margin-top: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="color: #d4a017; margin: 0; font-family: 'Inter', sans-serif; font-size: 16px; font-weight: 700;">
                        Outfit Look #{idx+1} (Score: {outfit['score']})
                    </h4>
                    <span style="font-size: 11px; font-weight: 600; color: #8a5e1a; background: rgba(200,146,42,0.12); border: 1px solid rgba(200,146,42,0.25); border-radius: 100px; padding: 2px 8px;">
                        {outfit['summary']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Render outfit items side-by-side
            items = outfit.get("items", [])
            item_cols = st.columns(len(items))
            for i, item in enumerate(items):
                with item_cols[i]:
                    st.markdown(f"""
                    <div style="text-align: center; font-size: 11px; font-weight: 700; color: #2b1b11; margin-top: 8px; margin-bottom: 4px;">
                        {item.get('category', '').upper()}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Catalog items do not have local photos, render custom stylized category icons
                    emoji = CATEGORY_EMOJIS.get(item.get("category", "").lower(), "👕")
                    st.markdown(f"""
                    <div style="background-color: #faf6f0; border: 1px dashed #dbb97a; border-radius: 10px; height: 120px; display: flex; align-items: center; justify-content: center; font-size: 38px; margin-bottom: 6px;">
                        {emoji}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="text-align: center; font-size: 10.5px; color: #5c4631; line-height: 1.2;">
                        <b>{item.get('name', '')}</b><br>
                        <span style="color: #8b7a6b;">({item.get('color_name', '')})</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Action bar below the outfit card
            col_space, col_save = st.columns([4, 1.2])
            with col_save:
                if st.button("💾 Save to Closet", key=f"save_hm_{idx}"):
                    success, save_msg = save_outfit(outfit, context_prompt, username)
                    if success:
                        st.success(save_msg)
                        if "saved_hm_indices" not in st.session_state:
                            st.session_state["saved_hm_indices"] = set()
                        st.session_state["saved_hm_indices"].add(idx)
                    else:
                        st.info(save_msg)
    else:
        st.info("Select occasion/weather details on the left and click 'Generate' to see H&M outfits here!")
