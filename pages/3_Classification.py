import streamlit as st
from PIL import Image
from model import load_model, predict

st.title("🌾 Rice Disease Classification")

disease_info = {
    "Healthy": "Leaf is healthy with no disease symptoms.",
    "Bacterial Blight": "Bacterial infection causing leaf blight.",
    "Blast": "Fungal disease causing blast spots.",
    "Brown Spot": "Fungal infection creating brown lesions.",
    "Tungro": "Virus-related disease affecting rice plants."
}

@st.cache_resource
def get_model():
    return load_model()

model_data = get_model()

uploaded_file = st.file_uploader("Upload Rice Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, use_column_width=True)

    with col2:
        if st.button("Predict Disease"):
            with st.spinner("Analyzing leaf..."):
                label, confidence, all_probs = predict(model_data, image)

            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence * 100:.2f}%")

            st.write("Prediction scores:")
            st.write({k: f"{v*100:.2f}%" for k, v in all_probs.items()})

            info = disease_info.get(label, "No information available.")
            st.write(info)
