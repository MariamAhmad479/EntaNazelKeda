import streamlit as st
from PIL import Image
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
    load_saved_outfits,
    remove_saved_outfit,
    get_vision_predictor,
    DATA_DIR
)

st.set_page_config(page_title="Wardrobe", page_icon="👕", layout="wide")
load_css()
render_sidebar()

# Fetch active user's Recommendation API
api = get_user_api()
username = st.session_state.get("username") or "guest"

# Display custom welcome message
if st.session_state.get("logged_in"):
    st.markdown(f"<h1>👕 {st.session_state['username'].capitalize()}'s Wardrobe</h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1>👕 My Wardrobe <span style='font-size: 16px; color: #8a7a6b;'>(Guest Account)</span></h1>", unsafe_allow_html=True)
st.write("Manage your closet items, upload new pieces via AI, and browse your saved outfits.")

# Tabs for organization
tab1, tab2 = st.tabs(["Upload & Manage Closet", "💾 Saved Outfits"])

# Helper for category emojis
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

with tab1:
    if st.session_state.get("upload_success_message"):
        st.success(st.session_state.upload_success_message)
        del st.session_state["upload_success_message"]

    col_upload, col_view = st.columns([1, 1.6], gap="medium")
    
    with col_upload:
        st.subheader("Upload New Items")
        st.write("Drop files here. Our Computer Vision AI will analyze their style, colors, and category automatically!")
        
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0
            
        uploaded_files = st.file_uploader(
            "Upload clothes images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}"
        )
        
        if uploaded_files:
            if st.button("Add to Wardrobe", key="add_items_btn"):
                # Ensure the user's images directory exists
                user_images_dir = os.path.join(DATA_DIR, "users", username, "images")
                os.makedirs(user_images_dir, exist_ok=True)
                
                # Retrieve the cached vision predictor
                predict_item_fn = get_vision_predictor()
                from populate_wardrobe import map_prediction_to_item
                
                added_count = 0
                
                with st.spinner("AI is analyzing your garment images..."):
                    for file in uploaded_files:
                        # Save the image to the user's images directory
                        img_path = os.path.join(user_images_dir, file.name)
                        with open(img_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        try:
                            # Run real machine learning predictions
                            image = Image.open(img_path)
                            prediction = predict_item_fn(image)
                            
                            # Skip innerwear items to keep wardrobe appropriate
                            if prediction.get("subCategory", "").lower() == "innerwear":
                                st.warning(f"Skipping innerwear item: {file.name}")
                                continue
                                
                            item_dict = map_prediction_to_item(prediction, file.name, image)
                            item_dict["image_path"] = img_path
                            
                            # Add to user's wardrobe file via API
                            api.add_item(item_dict)
                            added_count += 1
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {e}")
                            
                if added_count > 0:
                    st.session_state.upload_success_message = "Upload successful"
                    st.session_state.uploader_key += 1
                    st.rerun()
            
    with col_view:
        st.subheader("Your Actual Closet")
        wardrobe_items = api.get_wardrobe()
        
        if not wardrobe_items:
            st.info("Your wardrobe is empty. Upload some items to get started, or log in to view your personalized wardrobe!")
        else:
            # Dropdown category filter
            categories = ["All"] + sorted(list({item["category"].capitalize() for item in wardrobe_items}))
            selected_cat = st.selectbox("Filter closet by category:", categories)
            
            filtered_items = wardrobe_items
            if selected_cat != "All":
                filtered_items = [item for item in wardrobe_items if item["category"].capitalize() == selected_cat]
                
            if not filtered_items:
                st.info(f"No wardrobe items in the category: '{selected_cat}'")
            else:
                # Render in a clean 3-column grid
                cols = st.columns(3)
                for idx, item in enumerate(filtered_items):
                    col = cols[idx % 3]
                    with col:
                        # Card display using customizable styles
                        card_html = f"""
                        <div style="background-color: #f7edd8; border: 1.5px solid rgba(212,160,23,0.3); border-radius: 16px; padding: 12px; margin-bottom: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.03);">
                            <h4 style="color: #2b1b11; margin: 0 0 6px 0; font-size: 15px; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                {item['name']}
                            </h4>
                            <div style="font-size: 11px; color: #5c4631; line-height: 1.4;">
                                🏷️ <b>Category:</b> {item['category'].capitalize()}<br>
                                🎨 <b>Color:</b> {item['color_name'].capitalize()}<br>
                                🌡️ <b>Warmth:</b> {item['warmth_level']}/5
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                        
                        # Display photo or custom fallback
                        img_path = item.get("image_path")
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        else:
                            emoji = CATEGORY_EMOJIS.get(item["category"].lower(), "👕")
                            st.markdown(f"""
                            <div style="background-color: #f5edd8; border: 1px dashed #d4a017; border-radius: 12px; height: 180px; display: flex; align-items: center; justify-content: center; font-size: 60px; margin-bottom: 8px;">
                                {emoji}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if st.button("🗑️ Remove", key=f"del_{item['id']}"):
                            api.remove_item(item['id'])
                            st.success("Removed from closet!")
                            st.rerun()

with tab2:
    st.subheader("Your Saved Outfits")
    st.write("Here are the outfits you saved from the AI Stylist and H&M recommendations.")
    
    saved_outfits = load_saved_outfits(username)
    
    if not saved_outfits:
        st.warning("No saved outfits yet. Go to the AI Stylist or H&M Recommendations page to save some outfit suggestions!")
    else:
        for idx, outfit in enumerate(saved_outfits):
            st.markdown(f"""
            <div style="background-color: #fcf9f2; border: 2px solid #d4a017; border-radius: 16px; padding: 18px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h3 style="color: #d4a017; margin: 0; font-family: 'Caveat', cursive; font-size: 28px;">💖 Saved Outfit #{idx+1}</h3>
                    <span style="font-size: 12px; font-weight: 600; color: #8a5e1a; background: rgba(200,146,42,0.15); border: 1px solid rgba(200,146,42,0.3); border-radius: 100px; padding: 4px 12px;">
                        Score: {outfit.get('score', 'N/A')}
                    </span>
                </div>
                <p style="font-size: 13.5px; color: #5c4631; margin-bottom: 12px;"><b>Vibe / Prompt:</b> <i>"{outfit.get('context', 'Instant recommendation')}"</i></p>
                <div style="border-top: 1px dashed rgba(212,160,23,0.25); padding-top: 12px;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show items side-by-side
            items = outfit.get("items", [])
            item_cols = st.columns(len(items))
            for i, item in enumerate(items):
                with item_cols[i]:
                    st.markdown(f"""
                    <div style="text-align: center; font-size: 12px; font-weight: 700; color: #2b1b11; margin-bottom: 4px;">
                        {item.get('category', '').upper()}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    img_path = item.get("image_path")
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    else:
                        emoji = CATEGORY_EMOJIS.get(item.get("category", "").lower(), "👕")
                        st.markdown(f"""
                        <div style="background-color: #f0e6d2; border: 1px dashed #dbb97a; border-radius: 10px; height: 120px; display: flex; align-items: center; justify-content: center; font-size: 40px; margin-bottom: 6px;">
                            {emoji}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown(f"""
                    <div style="text-align: center; font-size: 11px; color: #6a5d4a; line-height: 1.2;">
                        <b>{item.get('name', '')}</b><br>
                        <span style="color: #8b7a6b;">({item.get('color_name', '')})</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Remove button for each saved outfit
            col_space, col_action = st.columns([5, 1])
            with col_action:
                if st.button("🗑️ Remove Outfit", key=f"del_outfit_{outfit.get('outfit_id', idx)}"):
                    remove_saved_outfit(outfit.get('outfit_id'), username)
                    st.success("Outfit removed!")
                    st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)