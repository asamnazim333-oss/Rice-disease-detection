import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🧪 Image Preprocessing")

uploaded_file = st.file_uploader(
    "Upload Rice Leaf Image",
    type=["jpg","png","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image)

    img = np.array(image)

    # Resize
    resized = cv2.resize(img, (224,224))

    # Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # Edge detection
    edges = cv2.Canny(blur,50,150)

    st.subheader("Preprocessing Steps")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(resized, caption="Resized Image")

    with col2:
        st.image(gray, caption="Grayscale")

    with col3:
        st.image(edges, caption="Edge Detection")
