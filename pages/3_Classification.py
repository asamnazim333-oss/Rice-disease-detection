import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("🤖 Rice Disease Classification (ResNet152)")

# Load model (cached)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("resnet152_model.h5")
    return model

model = load_model()

# Classes (must match model training)
classes = [
    "Healthy",
    "Brown spot",
    "Leaf blast"
]

def preprocess(image):
    img = image.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

uploaded = st.file_uploader("Upload Rice Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess(image)

    preds = model.predict(img)
    index = np.argmax(preds)
    confidence = np.max(preds) * 100

    st.success(f"Prediction: **{classes[index]}**")
    st.info(f"Confidence: {confidence:.2f}%")
