import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

st.title("🤖 Rice Disease Classification (ResNet152)")

# -------------------------------
# Model Download from Google Drive
# -------------------------------

MODEL_PATH = "resnet152_model.h5"
FILE_ID = "1sqkuE018KtBaKsgW2lmOEQ3GnLultn7y"

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait ⏳"):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------------
# Load Model
# -------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

st.success("Model loaded successfully ✅")

# -------------------------------
# Classes (must match training)
# -------------------------------

classes = [
    "Healthy",
    "Brown Spot",
    "Leaf Blast"
]

# -------------------------------
# Image Preprocessing
# -------------------------------

def preprocess(image):
    img = image.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------
# Image Upload
# -------------------------------

uploaded = st.file_uploader("Upload Rice Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess(image)

    with st.spinner("Predicting disease..."):
        preds = model.predict(img)

    index = np.argmax(preds)
    confidence = np.max(preds) * 100

    st.success(f"Prediction: **{classes[index]}**")
    st.info(f"Confidence: {confidence:.2f}%")

    st.write("Prediction probabilities:")
    st.write(preds)
