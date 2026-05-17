import streamlit as st
import json
import os
import hashlib
import shutil

# Resolve directories dynamically relative to this file's folder (Frontend)
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FRONTEND_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

USERS_FILE = os.path.join(DATA_DIR, "users.json")

def init_users_file():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    init_users_file()
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    
    if username in users:
        return False, "Username already exists."
    
    users[username] = hash_password(password)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)
        
    # Trigger initialization of their personal directory immediately
    get_user_api(username)
    
    return True, "Registered successfully!"

def authenticate_user(username, password):
    init_users_file()
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
        
    if username in users and users[username] == hash_password(password):
        return True
    return False

# ── Model Caching ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_nlp_model():
    """Lazily load and cache the NLP model to avoid reloading on every page rerun."""
    import sys
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    from nlp.inference import NLPInference
    return NLPInference()

@st.cache_resource
def get_vision_predictor():
    """Lazily load and cache the Computer Vision model predictor."""
    import sys
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    from vision.predict import predict_item
    return predict_item

# ── User Directory & API Isolation ────────────────────────────────────────────
def get_user_api(username=None):
    """Retrieve or initialize the RecommendationAPI instance for a given user.
    If no user is logged in, falls back to the 'guest' user directory.
    Clones data/sample_wardrobe.json as their template closet if wardrobe.json doesn't exist yet.
    """
    if not username:
        username = st.session_state.get("username") or "guest"
        
    # Setup paths under data/users/<username>/
    user_dir = os.path.join(DATA_DIR, "users", username)
    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(os.path.join(user_dir, "images"), exist_ok=True)
    
    wardrobe_path = os.path.join(user_dir, "wardrobe.json")
    feedback_path = os.path.join(user_dir, "feedback_log.json")
    
    # Initialize empty wardrobe/feedback log if the files do not exist
    if not os.path.exists(wardrobe_path):
        with open(wardrobe_path, 'w', encoding='utf-8') as f:
            json.dump({"items": []}, f)
            
    if not os.path.exists(feedback_path):
        with open(feedback_path, 'w', encoding='utf-8') as f:
            json.dump({}, f)
                
    import sys
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    from recommendation_engine.api import RecommendationAPI
    
    return RecommendationAPI(wardrobe_path=wardrobe_path, feedback_path=feedback_path)

# ── Saved Outfits Operations ──────────────────────────────────────────────────
def load_saved_outfits(username=None):
    """Load a user's saved outfits from their personal saved_outfits.json file."""
    if not username:
        username = st.session_state.get("username") or "guest"
    outfits_path = os.path.join(DATA_DIR, "users", username, "saved_outfits.json")
    if os.path.exists(outfits_path):
        try:
            with open(outfits_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_outfit(outfit_dict, context_prompt="Recommendation", username=None):
    """Save an outfit to a user's personal saved_outfits.json file."""
    if not username:
        username = st.session_state.get("username") or "guest"
    user_dir = os.path.join(DATA_DIR, "users", username)
    os.makedirs(user_dir, exist_ok=True)
    outfits_path = os.path.join(user_dir, "saved_outfits.json")
    
    outfits = []
    if os.path.exists(outfits_path):
        try:
            with open(outfits_path, 'r', encoding='utf-8') as f:
                outfits = json.load(f)
        except Exception:
            outfits = []
            
    # Check if outfit already saved by comparing keys or ids
    outfit_id = outfit_dict.get("outfit_id")
    if any(o.get("outfit_id") == outfit_id for o in outfits):
        return False, "Outfit already saved!"
        
    # Append context details
    outfit_to_save = dict(outfit_dict)
    outfit_to_save["context"] = context_prompt
    outfits.append(outfit_to_save)
    
    try:
        with open(outfits_path, 'w', encoding='utf-8') as f:
            json.dump(outfits, f, indent=2)
        return True, "Outfit saved successfully!"
    except Exception as e:
        return False, f"Failed to save outfit: {e}"

def remove_saved_outfit(outfit_id, username=None):
    """Remove a saved outfit from the user's personal saved_outfits.json file."""
    if not username:
        username = st.session_state.get("username") or "guest"
    outfits_path = os.path.join(DATA_DIR, "users", username, "saved_outfits.json")
    if os.path.exists(outfits_path):
        try:
            with open(outfits_path, 'r', encoding='utf-8') as f:
                outfits = json.load(f)
            outfits = [o for o in outfits if o.get("outfit_id") != outfit_id]
            with open(outfits_path, 'w', encoding='utf-8') as f:
                json.dump(outfits, f, indent=2)
            return True
        except Exception:
            return False
    return False

# ── General CSS and Sidebar Rendering ─────────────────────────────────────────
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ================= MAIN APP (Lighter Beige Theme) ================= */
    .stApp {
        background-color: #faf6f0; /* Light cream/beige */
        color: #2b1b11;
    }

    /* ================= REMOVE HEADER ================= */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stSidebarCollapseButton"] { 
        display: none !important; 
    }
    [data-testid="collapsedControl"] { 
        display: none !important; 
    }
    /* ================= SIDEBAR ================= */
    section[data-testid="stSidebar"] {
        background-color: #f0e6d2;
        border-right: 1px solid rgba(212,160,23,0.3);
    }
    section[data-testid="stSidebar"] * {
        color: #2b1b11 !important;
        font-family: 'Inter', sans-serif;
    }

    /* ================= FONTS ================= */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #3d2a1b !important;
    }
    p, div, span, label {
        font-family: 'Inter', sans-serif;
        color: #3d2a1b;
    }

    /* ================= BUTTON ================= */
    .stButton > button {
        width: 100%;
        background-color: #d4a017;
        color: #ffffff !important;
        border: none;
        border-radius: 12px;
        padding: 12px;
        font-size: 16px;
        font-weight: 600;
        transition: 0.3s;
        box-shadow: 0 4px 10px rgba(212,160,23,0.2);
    }
    .stButton > button:hover {
        background-color: #e5b32e;
        transform: translateY(-2px);
    }
    .stButton > button:disabled {
        background-color: #e3cca6;
        color: #8c7b6b !important;
        opacity: 1;
        cursor: default;
        transform: none;
        box-shadow: none;
    }

    /* ================= CARDS (Beige) ================= */
    .card {
        background-color: #f7edd8; /* Beige card */
        border: 1px solid rgba(212,160,23,0.4);
        border-radius: 16px;
        padding: 25px 20px;
        text-align: center;
        min-height: 220px;
        transition: 0.3s;
        text-decoration: none;
        display: block;
        color: #3d2a1b;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .card:hover {
        transform: translateY(-6px);
        border: 1px solid #d4a017;
        box-shadow: 0 8px 16px rgba(212,160,23,0.15);
    }
    .card-icon {
        font-size: 40px;
        margin-bottom: 12px;
    }
    .card-title {
        font-size: 22px;
        font-weight: bold;
        color: #2b1b11;
        margin-bottom: 10px;
    }
    .card-text {
        color: #5c4631;
        line-height: 1.6;
        font-size: 15px;
    }
    
    /* ================= CHAT ================= */
    .stChatMessage {
        background-color: #f0e6d2; /* Light beige for chat messages */
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }
    
    /* Hide the default file uploader text if we use it directly */
    .uploadedFile {
        color: #3d2a1b !important;
    }

    /* Typography overrides for Home */
    .home-title {
        text-align: center; 
        font-size: 52px; 
        color: #2b1b11; 
        margin-top: 0;
        font-weight: 800;
    }
    .home-title span {
        color: #d4a017;
    }
    .home-subtitle {
        text-align: center; 
        color: #d4a017; 
        font-size: 22px; 
        font-weight: 600; 
        margin-bottom: 20px;
    }
    .home-desc {
        text-align: center; 
        font-size: 18px; 
        color: #5c4631; 
        line-height: 1.6; 
        max-width: 800px; 
        margin: 0 auto 40px auto;
    }

    </style>
    """, unsafe_allow_html=True)

@st.dialog("Authentication")
def auth_dialog():
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login")
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="login_btn"):
            if authenticate_user(login_user, login_pass):
                st.session_state["username"] = login_user
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("Invalid credentials.")
                
    with tab2:
        st.subheader("Sign Up")
        reg_user = st.text_input("Username", key="reg_user")
        reg_pass = st.text_input("Password", type="password", key="reg_pass")
        reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
        if st.button("Sign Up", key="reg_btn"):
            if reg_pass != reg_confirm:
                st.error("Passwords do not match.")
            elif not reg_user or not reg_pass:
                st.error("Fields cannot be empty.")
            else:
                success, msg = register_user(reg_user, reg_pass)
                if success:
                    st.success(msg)
                    st.session_state["username"] = reg_user
                    st.session_state["logged_in"] = True
                    st.rerun()
                else:
                    st.error(msg)

def render_sidebar():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = None

    with st.sidebar:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.session_state["logged_in"]:
            st.button(f"👤 {st.session_state['username']}", disabled=True)
            if st.button("🚪 Logout"):
                st.session_state["logged_in"] = False
                st.session_state["username"] = None
                # Clear chatbot session to reload for next login
                if "chatbot" in st.session_state:
                    del st.session_state["chatbot"]
                if "chatbot_username" in st.session_state:
                    del st.session_state["chatbot_username"]
                st.rerun()
        else:
            if st.button("Login / Sign Up"):
                auth_dialog()
