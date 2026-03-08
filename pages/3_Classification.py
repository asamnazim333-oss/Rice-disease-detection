import streamlit as st
import numpy as np
from PIL import Image
from model import load_model, predict

st.title("🌾 Rice Disease Classification")

# Disease info dictionary (define it early)
disease_info = {
    "Healthy": "Leaf is healthy with no disease symptoms.",
    "Bacterial_Leaf_Blight": "Caused by bacteria Xanthomonas.",
    "Leaf_Blast": "Fungal disease caused by Magnaporthe oryzae.",
    "Brown_Spot": "Fungal infection causing brown lesions."
}

# Load model once
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

uploaded_file = st.file_uploader(
    "Upload Rice Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)

    with col2:

        if st.button("Predict Disease"):

            with st.spinner("Analyzing leaf..."):

                img = image.resize((224,224))
                img = np.array(img) / 255.0

                label, confidence = predict(model, img)

            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence * 100:.2f}%")
            st.progress(float(confidence))

            # Show disease info only if exists
            info = disease_info.get(label, "No information available.")
            st.write(info)
