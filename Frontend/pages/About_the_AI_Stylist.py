import streamlit as st
import sys
import os

# Ensure the parent directory is in the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_css, render_sidebar

st.set_page_config(page_title="About the AI Stylist", page_icon="ℹ️", layout="wide")
load_css()
render_sidebar()

st.title("ℹ️ About the AI Stylist")

st.write("""
**Enta Nazel Keda?** is an AI stylist app designed to help users choose suitable outfits
based on different factors such as occasion, weather, style preference, and uploaded clothing items.
""")

st.markdown("### Main idea")

st.write("""
The app uses user input and clothing images to recommend outfit combinations.
It can later be connected to a machine learning model for item classification and personalized styling.
""")

st.markdown("### Future AI features")

st.write("""
- Clothing item detection
- Color matching
- Weather-based outfit suggestions
- Occasion-based recommendations
- Personalized recommendations based on saved outfits
""")