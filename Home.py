import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Enta Nazel Keda?",
    page_icon="👕",
    layout="wide"
)

def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background: radial-gradient(circle at top, #3a2818 0%, #231a0f 45%, #140d07 100%);
        color: #f3e4cf;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f3e4cf 0%, #fff4df 100%);
        border-right: 1px solid rgba(228,166,33,0.25);
    }

    section[data-testid="stSidebar"] * {
        color: #231a0f !important;
        font-family: 'Inter', sans-serif;
    }

    div[data-testid="stSidebarNav"] ul {
        padding-top: 20px;
    }

    div[data-testid="stSidebarNav"] a {
        border-radius: 10px;
        margin: 6px 12px;
        padding: 12px 14px;
        font-weight: 600;
    }

    div[data-testid="stSidebarNav"] a:hover {
        background-color: rgba(228,166,33,0.25);
    }

    div[data-testid="stSidebarNav"] a[aria-current="page"] {
        background: linear-gradient(135deg, #e4a621, #c98613);
        color: #231a0f !important;
        box-shadow: 0 8px 20px rgba(228,166,33,0.25);
    }

    h1, h2, h3 {
        font-family: 'Cinzel', serif;
    }

    p, div, span, label {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-family: 'Cinzel', serif;
        font-size: 76px;
        font-weight: 700;
        color: #f3e4cf;
        text-align: center;
        margin-bottom: 5px;
        text-shadow: 0px 3px 18px rgba(0,0,0,0.5);
    }

    .gold-text {
        color: #e4a621;
    }

    .subtitle {
        text-align: center;
        color: #e4a621;
        font-size: 24px;
        margin-bottom: 20px;
        font-weight: 500;
    }

    .description {
        text-align: center;
        color: #f3e4cf;
        font-size: 18px;
        line-height: 1.8;
        max-width: 700px;
        margin: auto;
    }

    .divider {
        text-align: center;
        color: #e4a621;
        font-size: 26px;
        margin: 20px 0;
    }

    .hero-logo {
        display: flex;
        justify-content: center;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .hero-logo img {
        width: 130px;
        height: 130px;
        object-fit: contain;
        border-radius: 50%;
        border: 3px solid #e4a621;
        padding: 12px;
        background-color: rgba(243,228,207,0.08);
        box-shadow: 0 0 30px rgba(228,166,33,0.25);
    }

    .stButton > button {
        background: linear-gradient(135deg, #e4a621, #c98613);
        color: #231a0f;
        border: none;
        border-radius: 12px;
        padding: 13px 35px;
        font-size: 18px;
        font-weight: 700;
        box-shadow: 0 8px 24px rgba(228,166,33,0.28);
    }

    .stButton > button:hover {
        background: #f3e4cf;
        color: #231a0f;
        border: none;
    }

    .card {
        background: rgba(243,228,207,0.06);
        border: 1px solid rgba(228,166,33,0.30);
        border-radius: 18px;
        padding: 32px 22px;
        text-align: center;
        min-height: 230px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.22);
    }

    .card-icon {
        font-size: 42px;
        color: #e4a621;
        margin-bottom: 15px;
    }

    .card-title {
        font-family: 'Cinzel', serif;
        font-size: 25px;
        color: #f3e4cf;
        margin-bottom: 10px;
        font-weight: 600;
    }

    .card-text {
        color: #dbc8ad;
        font-size: 15px;
        line-height: 1.7;
    }

    .sidebar-logo {
        text-align: center;
        padding: 30px 10px 20px 10px;
    }

    .sidebar-logo img {
        width: 140px;
        height: 140px;
        object-fit: contain;
    }

    .sidebar-brand {
        font-family: 'Cinzel', serif;
        font-size: 24px;
        font-weight: 700;
        color: #231a0f;
        text-align: center;
        letter-spacing: 1px;
    }

    .sidebar-subtitle {
        color: #e4a621;
        font-size: 13px;
        text-align: center;
        letter-spacing: 5px;
        margin-top: 6px;
        font-weight: 700;
    }

    .sidebar-note {
        background-color: rgba(228,166,33,0.12);
        border-radius: 15px;
        padding: 20px;
        margin: 30px 10px;
        border-left: 4px solid #e4a621;
        font-size: 14px;
        line-height: 1.6;
    }

    .footer {
        text-align: center;
        color: #e4a621;
        margin-top: 50px;
        font-size: 15px;
    }

    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# ---------- SIDEBAR LOGO ----------
with st.sidebar:
    st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
    st.image("data/logo.png", use_container_width=False, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-brand">Enta Nazel Keda?</div>
    <div class="sidebar-subtitle">AI STYLIST</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-note">
        ✨ Your personal AI stylist for every occasion.
    </div>
    """, unsafe_allow_html=True)

# ---------- MAIN PAGE ----------
st.markdown("""
<div class="hero-logo">
    <img src="app/staticd/logo.png">
</div>
""", unsafe_allow_html=True)

# This displays the logo correctly from local Streamlit
st.image("data/logo.png", width=120)

st.markdown("""
<h1 class="main-title">Enta <span class="gold-text">Nazel</span> Keda?</h1>
<div class="subtitle">Your AI Stylist Assistant</div>
<div class="divider">━━━━ ✦ ━━━━</div>
<p class="description">
Get outfit recommendations tailored to your style, occasion, and the weather.
Look your best, every day.
</p>
""", unsafe_allow_html=True)

st.write("")
st.write("")

center_col1, center_col2, center_col3 = st.columns([1, 1, 1])

with center_col2:
    if st.button("✨ Get Started"):
        st.switch_page("pages/1_Upload_Outfit.py")

st.write("")
st.write("")
st.write("")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-icon">☁️</div>
        <div class="card-title">Upload</div>
        <div class="card-text">
            Upload your clothes and build your personal wardrobe.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-icon">✨</div>
        <div class="card-title">Recommend</div>
        <div class="card-text">
            Get AI-powered outfit recommendations instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-icon">♡</div>
        <div class="card-title">Save</div>
        <div class="card-text">
            Save your favorite outfits and access them anytime.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
        <div class="card-icon">ⓘ</div>
        <div class="card-title">Learn</div>
        <div class="card-text">
            Learn how our AI stylist helps you look your best.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    ♥ Made with love for your style ♥
</div>
""", unsafe_allow_html=True)