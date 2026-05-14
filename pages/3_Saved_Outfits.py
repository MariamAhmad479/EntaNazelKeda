import streamlit as st

st.title("💾 Saved Outfits")

st.write("Here you will see outfits you saved.")

if "saved_outfits" not in st.session_state:
    st.session_state.saved_outfits = []

if len(st.session_state.saved_outfits) == 0:
    st.warning("No saved outfits yet.")
else:
    for outfit in st.session_state.saved_outfits:
        st.write(outfit)