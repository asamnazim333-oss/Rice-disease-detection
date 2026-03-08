import streamlit as st
import numpy as np
from PIL import Image

st.title("Model Explainability (Grad-CAM)")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Original Image")

    st.write("Grad-CAM will highlight disease regions on the leaf.")

    st.warning("Demo version - highlight areas where model focuses.")
