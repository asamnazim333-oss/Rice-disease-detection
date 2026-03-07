import streamlit as st
from PIL import Image
import numpy as np

st.title("🧹 Preprocessing")

st.write("""
- Resize images to 224x224  
- Normalize pixel values to 0-1  
- Convert to array for model input
""")

uploaded = st.file_uploader("Upload image to see preprocessing", type=["jpg", "png", "jpeg"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Original Image", use_column_width=True)

    img_resized = image.resize((224, 224))
    st.image(img_resized, caption="Resized 224x224", use_column_width=True)

    img_array = np.array(img_resized)/255.0
    st.write("Normalized array shape:", img_array.shape)
