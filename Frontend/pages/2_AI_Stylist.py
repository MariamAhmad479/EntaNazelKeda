import streamlit as st
import time
import sys
import os
from PIL import Image

# Ensure parent directory is in path to import utils
FRONTEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(FRONTEND_DIR)
sys.path.append(FRONTEND_DIR)
sys.path.append(PROJECT_ROOT)

from utils import (
    load_css,
    render_sidebar,
    get_user_api,
    get_nlp_model,
    get_vision_predictor,
    save_outfit,
    DATA_DIR
)

st.set_page_config(page_title="AI Stylist", page_icon="✨", layout="wide")
load_css()
render_sidebar()

st.title("✨ AI Stylist Chat")
st.write("Chat naturally with your digital fashion consultant. Try saying things like *'I need a casual look for hot weather'* or upload a picture of a garment to instantly add it to your wardrobe and get matching advice!")

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

# ── Chatbot State Management ──────────────────────────────────────────────────
# Resolve API and re-initialize dialogue manager if username changes or chatbot doesn't exist
if "chatbot_username" not in st.session_state or st.session_state.chatbot_username != st.session_state.get("username"):
    with st.spinner("Initializing AI Dialogue Engine..."):
        api = get_user_api()
        nlp_model = get_nlp_model()
        from main import WardrobeChatbot
        st.session_state.chatbot = WardrobeChatbot(api, nlp_model)
        st.session_state.chatbot_username = st.session_state.get("username")
        
        # Reset chat history for a clean state
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"Hello! I am your Smart AI Stylist, bound to your personal wardrobe. "
                           f"Where are you heading today, and what is the weather like? "
                           f"I'll curate the absolute best outfit from your collection! "
                           f"(Type *'shop'* to explore H&M global catalog recommendations instead!)"
            }
        ]

# ── Render Message History ────────────────────────────────────────────────────
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display image upload context if user uploaded an image in this turn
        if "user_image" in message:
            try:
                st.image(message["user_image"], width=200)
            except Exception:
                pass
                
        # Render visual outfits if recommendations were generated in this turn
        if "outfits" in message:
            outfits = message["outfits"]
            context_prompt = message.get("context", "Style Advice")
            
            st.markdown("<p style='font-weight:600; color:#d4a017; font-size:15px; margin: 15px 0 5px 0;'>✨ Visual Recommendations:</p>", unsafe_allow_html=True)
            
            for o_idx, outfit in enumerate(outfits):
                # Score banner
                st.markdown(f"""
                <div style="background-color: #f7edd8; border: 1.5px solid rgba(212,160,23,0.3); border-radius: 12px 12px 0 0; padding: 10px 14px; margin-top: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #2b1b11; font-size: 14.5px;">Outfit Option #{o_idx+1} (Score: {outfit['score']})</span>
                        <span style="font-size: 11px; font-weight: 500; color: #8a5e1a; background: rgba(200,146,42,0.1); border: 1px solid rgba(200,146,42,0.25); border-radius: 100px; padding: 2px 8px;">
                            {outfit['summary']}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Grid wrapper for items
                items = outfit.get("items", [])
                item_cols = st.columns(len(items))
                for item_idx, item in enumerate(items):
                    with item_cols[item_idx]:
                        # Style container
                        st.markdown(f"""
                        <div style="text-align: center; font-size: 11px; font-weight: 700; color: #2b1b11; margin-top: 8px; margin-bottom: 4px;">
                            {item.get('category', '').upper()}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Real picture or emoji
                        img_path = item.get("image_path")
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        else:
                            emoji = CATEGORY_EMOJIS.get(item.get("category", "").lower(), "👕")
                            st.markdown(f"""
                            <div style="background-color: #faf6f0; border: 1px dashed #dbb97a; border-radius: 8px; height: 100px; display: flex; align-items: center; justify-content: center; font-size: 34px; margin-bottom: 4px;">
                                {emoji}
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Attributes
                        st.markdown(f"""
                        <div style="text-align: center; font-size: 10px; color: #5c4631; line-height: 1.2;">
                            <b>{item.get('name', '')}</b><br>
                            <span style="color:#8b7a6b;">({item.get('color_name', '')})</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Card footer with Save Outfit button
                col_left, col_save = st.columns([4, 1.2])
                with col_save:
                    if st.button("💾 Save Option", key=f"save_chat_{idx}_{o_idx}"):
                        success, save_msg = save_outfit(outfit, context_prompt, st.session_state.chatbot_username)
                        if success:
                            st.success(save_msg)
                        else:
                            st.info(save_msg)
                st.markdown("<div style='border-bottom: 1.5px solid rgba(212,160,23,0.25); margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# ── Handle User Input ─────────────────────────────────────────────────────────
try:
    chat_data = st.chat_input("Message AI Stylist or type commands like 'shop'...", accept_file=["png", "jpg", "jpeg"])
except TypeError:
    chat_data = None
    col1, col2 = st.columns([1, 15])
    with col1:
        with st.popover("➕", help="Upload a garment photo"):
            uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    with col2:
        prompt = st.chat_input("Message AI Stylist or type commands like 'shop'...")
        if prompt:
            chat_data = {"text": prompt, "files": [uploaded_file] if uploaded_file else []}

if chat_data:
    if isinstance(chat_data, dict):
        prompt = chat_data.get("text", "")
        files = chat_data.get("files", [])
        uploaded_file = files[0] if files else None
    else:
        prompt = chat_data
        uploaded_file = None

    if prompt or uploaded_file is not None:
        # Display user message in chat
        with st.chat_message("user"):
            if prompt:
                st.markdown(prompt)
            if uploaded_file is not None:
                user_img = Image.open(uploaded_file)
                st.image(user_img, caption="Uploaded Garment", width=200)

        # Save to chat history
        user_msg = {"role": "user", "content": prompt}
        if uploaded_file is not None:
            user_msg["user_image"] = Image.open(uploaded_file)
        st.session_state.messages.append(user_msg)

        # ── AI Stylist Processing ──
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Case 1: An image was uploaded in this dialogue turn
            if uploaded_file is not None:
                user_images_dir = os.path.join(DATA_DIR, "users", st.session_state.chatbot_username or "guest", "images")
                os.makedirs(user_images_dir, exist_ok=True)
                img_path = os.path.join(user_images_dir, uploaded_file.name)
                
                with open(img_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                with st.spinner("Scanning image with Computer Vision model..."):
                    try:
                        # Scan using real CV predictor
                        predict_item_fn = get_vision_predictor()
                        from populate_wardrobe import map_prediction_to_item
                        
                        image = Image.open(img_path)
                        prediction = predict_item_fn(image)
                        
                        if prediction.get("subCategory", "").lower() == "innerwear":
                            response_text = "I analyzed the image you uploaded, but it appears to be innerwear. I have skipped adding it to protect closet quality! Try uploading outerwear garments like jackets, shirts, or pants."
                        else:
                            # Map prediction metadata and add to closet
                            item_dict = map_prediction_to_item(prediction, uploaded_file.name, image)
                            item_dict["image_path"] = img_path
                            
                            # Add to closet JSON
                            api = get_user_api(st.session_state.chatbot_username)
                            api.add_item(item_dict)
                            
                            color = prediction.get("baseColour", "unknown").capitalize()
                            article = prediction.get("articleType", "item").capitalize()
                            season = prediction.get("season", "any season").capitalize()
                            occasion = prediction.get("usage", "casual").capitalize()
                            
                            response_text = (
                                f"✨ **Garment Added Successfully!** ✨\n\n"
                                f"I scanned your photo and successfully added it to your closet!\n"
                                f"- **Detected Piece:** {color} {article}\n"
                                f"- **Best Seasons:** {season}\n"
                                f"- **Occasion Code:** {occasion}\n\n"
                                f"Would you like me to suggest an outfit that incorporates this newly added piece?"
                            )
                    except Exception as e:
                        response_text = f"I ran into an error scanning that picture: {e}. Please try another format or image!"
            
            # Case 2: A text query was entered
            else:
                with st.spinner("AI is thinking..."):
                    try:
                        response_text = st.session_state.chatbot.handle_input(prompt)
                    except Exception as e:
                        response_text = f"Sorry, I ran into an issue parsing your query: {e}. Try checking your NLP training model."

            # Typing simulation stream
            full_response = ""
            for chunk in response_text.split():
                full_response += chunk + " "
                time.sleep(0.04)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
            # Check if chatbot generated outfit recommendations in this turn
            assistant_msg = {"role": "assistant", "content": full_response}
            if st.session_state.chatbot.last_recommendations:
                assistant_msg["outfits"] = st.session_state.chatbot.last_recommendations
                assistant_msg["context"] = prompt if prompt else "CV Upload matching"
                # Clear recommendations once captured
                st.session_state.chatbot.last_recommendations = []
                
            st.session_state.messages.append(assistant_msg)
            st.rerun()