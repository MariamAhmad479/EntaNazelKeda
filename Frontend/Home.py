import streamlit as st
import base64
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_css, render_sidebar

st.set_page_config(
    page_title="Enta Nazel Keda?",
    page_icon="✨",
    layout="wide"
)

load_css()
render_sidebar()

# ── Logo ──────────────────────────────────────────────────────────────────────
logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logo.jpeg")
with open(logo_path, "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&family=Sora:wght@300;400;600&display=swap');

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main { background-color: #1a1610 !important; }

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background-color: #f0e8d8 !important;
}

#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

.block-container {
    padding: 2.5rem 2.5rem 4rem !important;
    max-width: 860px !important;
    margin: 0 auto !important;
}

/* ── Hero ── */
.enk-hero {
    display: flex; flex-direction: column;
    align-items: center; text-align: center;
    margin-bottom: 2rem;
}
.enk-logo {
    width: 118px; height: 118px;
    border-radius: 24px;
    border: 2.5px solid #c8922a;
    padding: 4px; background: #211d14;
    box-shadow: 0 0 40px rgba(200,146,42,0.2);
    margin-bottom: 1.3rem; overflow: hidden;
}
.enk-logo img {
    width: 100%; height: 100%;
    object-fit: cover; border-radius: 20px; display: block;
}
.enk-eyebrow {
    font-family: 'Sora', sans-serif;
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.26em; color: #c8922a;
    text-transform: uppercase; margin-bottom: 0.4rem;
}
.enk-title {
    font-family: 'Caveat', cursive;
    font-size: 80px; font-weight: 700;
    line-height: 1.0; color: #f0e8d8 !important;
    margin: 0 0 0.15em;
}
.enk-title .gold { color: #e8a825 !important; }
.enk-sub {
    font-family: 'Sora', sans-serif;
    font-size: 10px; font-weight: 300;
    letter-spacing: 0.24em; text-transform: uppercase;
    color: #6a5d4a; margin-bottom: 0.9rem;
}
.enk-desc {
    font-family: 'Sora', sans-serif;
    font-size: 14.5px; font-weight: 300;
    color: #9a8878; max-width: 480px; line-height: 1.8;
}
.enk-divider {
    width: 280px; height: 1px;
    background: linear-gradient(90deg, transparent, #c8922a55, #c8922aaa, #c8922a55, transparent);
    margin: 1.8rem auto 2.4rem;
}
/* ── Cards with button inside ── */
.enk-card {
    background: #f5ede0;
    border: 1.5px solid #dbb97a;
    border-radius: 20px;
    padding: 1.8rem 1.3rem 1.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(200,146,42,0.08);
    display: flex;
    flex-direction: column;
    align-items: center;
}
.enk-card-top {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: #c8922a;
    border-radius: 20px 20px 0 0;
}
.enk-card-icon {
    font-size: 36px; margin-bottom: 0.75rem;
    display: block; line-height: 1;
}
.enk-card-title {
    font-family: 'Caveat', cursive;
    font-size: 28px; font-weight: 700;
    color: #2b1b11; margin-bottom: 0.4rem;
}
.enk-card-tag {
    display: inline-block;
    font-family: 'Sora', sans-serif;
    font-size: 8px; font-weight: 600;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #8a5e1a;
    background: rgba(200,146,42,0.15);
    border: 1px solid rgba(200,146,42,0.35);
    border-radius: 100px; padding: 3px 10px;
    margin-bottom: 0.8rem;
}
.enk-card-desc {
    font-family: 'Sora', sans-serif;
    font-size: 12.5px; font-weight: 300;
    color: #5c4631; line-height: 1.65;
    margin-bottom: 1.2rem;
    flex-grow: 1;
}
.enk-card-btn {
    display: inline-block;
    font-family: 'Sora', sans-serif;
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #fff !important;
    background: #6a4e23 !important; /* Brown color */
    border-radius: 50px;
    padding: 8px 22px;
    text-decoration: none !important;
    border: none;
    cursor: pointer;
    transition: 0.3s;
}
.enk-card-btn:hover {
    background: #4a3516 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="enk-hero">
    <div class="enk-logo">
        <img src="data:image/jpeg;base64,{logo_b64}" alt="Logo" />
    </div>
    <p class="enk-eyebrow">Cairo's AI Fashion Assistant</p>
    <h1 class="enk-title">Enta <span class="gold">Nazel</span> Keda?</h1>
    <p class="enk-sub">Luxury AI Styling Experience</p>
    <p class="enk-desc">
        Your personal AI stylist that builds outfits from your wardrobe,
        reads your mood, and knows what fits the occasion — even the weather.
    </p>
</div>
<div class="enk-divider"></div>
""", unsafe_allow_html=True)

# ── Cards — each card has its button baked in ─────────────────────────────────
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
    <div class="enk-card">
        <div class="enk-card-top"></div>
        <span class="enk-card-icon">👕</span>
        <div class="enk-card-title">Wardrobe</div>
        <div class="enk-card-tag">Upload &amp; Manage</div>
        <p class="enk-card-desc">Upload your pieces and build your digital closet. Browse saved looks anytime.</p>
        <a href="/1_Wardrobe" target="_self" class="enk-card-btn">Open Wardrobe</a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="enk-card">
        <div class="enk-card-top"></div>
        <span class="enk-card-icon">🧠</span>
        <div class="enk-card-title">AI Stylist</div>
        <div class="enk-card-tag">Powered by AI</div>
        <p class="enk-card-desc">Chat with the AI to get outfit recommendations tailored to your day and vibe.</p>
        <a href="/2_AI_Stylist" target="_self" class="enk-card-btn">Start Styling</a>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="enk-card">
        <div class="enk-card-top"></div>
        <span class="enk-card-icon">💡</span>
        <div class="enk-card-title">About</div>
        <div class="enk-card-tag">How it Works</div>
        <p class="enk-card-desc">Learn how the AI reads colors, textures, and trends to style you perfectly.</p>
        <a href="/About_the_AI_Stylist" target="_self" class="enk-card-btn">Learn More</a>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; font-family:'Sora',sans-serif; font-size:11px;
            font-weight:300; color:#3d3629; letter-spacing:0.14em; margin-top:2.5rem;">
    <span style="color:#c8922a;">♥</span>
    Designed with elegance for modern fashion lovers
    <span style="color:#c8922a;">♥</span>
</div>
""", unsafe_allow_html=True)