import streamlit as st

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
    st.success("Recommended Outfit")

    st.write("👕 Top: White shirt")
    st.write("👖 Bottom: Black pants")
    st.write("👟 Shoes: White sneakers")
    st.write("🧥 Extra: Light jacket")

    st.info(
        "This outfit matches your selected occasion, weather, and preferred style."
    )