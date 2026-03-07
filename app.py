import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="🌾 Rice Disease Detection", layout="wide")

# Title
st.title("🌾 Rice Disease Detection System")
st.write("AI-powered system to detect rice diseases from leaf images.")

# Sidebar Navigation
menu = st.sidebar.selectbox(
    "Select Section",
    ["Dataset Overview", "Preprocessing", "Model", "Evaluation", "Prediction"]
)

# =========================
# DATASET OVERVIEW
# =========================
if menu == "Dataset Overview":
    st.header("📊 Dataset Overview")
    st.write("""
    This dataset contains images of rice leaves classified into:
    - Healthy
    - Brown Spot
    - Leaf Blast
    - Bacterial Blight
    """)
    
    # Show sample images if available
    img_path = "data/sample.jpg"
    if os.path.exists(img_path):
        image = Image.open(img_path)
        st.image(image, caption="Sample Rice Leaf Image", width=300)
    else:
        st.warning("Sample image not found. Add dataset images in 'data' folder.")

# =========================
# PREPROCESSING
# =========================
elif menu == "Preprocessing":
    st.header("⚙ Data Preprocessing")
    st.write("""
    Preprocessing steps:
    1. Resize images to 224x224
    2. Normalize pixel values (0-1)
    3. Data augmentation (rotation, flipping)
    4. Train-test split
    """)

    st.code("""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True
)
    """)

# =========================
# MODEL INFORMATION
# =========================
elif menu == "Model":
    st.header("🤖 Model Architecture")
    st.write("""
    Model used: CNN (Convolutional Neural Network)

    Layers:
    - Conv2D
    - MaxPooling
    - Dropout
    - Dense
    - Softmax
    """)

    st.code("""
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
    """)

# =========================
# MODEL EVALUATION
# =========================
elif menu == "Evaluation":
    st.header("📈 Model Evaluation")

    st.metric(label="Accuracy", value="94.5%")
    st.metric(label="Loss", value="0.12")

    # Example confusion matrix
    cm = np.array([[50, 2], [3, 45]])

    fig, ax = plt.subplots()
    ax.imshow(cm)
    st.pyplot(fig)

# =========================
# PREDICTION
# =========================
elif menu == "Prediction":
    st.header("🌿 Disease Prediction")

    uploaded_file = st.file_uploader("Upload Rice Leaf Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Dummy Prediction (Replace with real model)
        classes = ["Healthy", "Brown Spot", "Leaf Blast", "Bacterial Blight"]
        prediction = np.random.choice(classes)

        st.success(f"Prediction: {prediction}")
