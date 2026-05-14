import streamlit as st
from PIL import Image
from vision import predict_item

st.title("Upload Outfit")

uploaded_file = st.file_uploader(
    "Upload clothing image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, width=300)

    with st.spinner("Analyzing outfit..."):
        prediction = predict_item(image)

    st.success("Prediction Complete")

    st.json(prediction)

else:
    st.info("Please upload an image.")