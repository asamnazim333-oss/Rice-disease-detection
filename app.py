import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

st.set_page_config(page_title="🌾 Rice Disease Detection")

st.title("🌾 Rice Disease Detection App")
st.write("Upload rice leaf image to detect disease")

# Load pretrained model
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Class labels (example only — adjust based on your dataset)
classes = ["Healthy", "Bacterial Blight", "Blast", "Brown Spot"]

def predict(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    return classes[pred % len(classes)]  # simple mapping

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    result = predict(image)
    st.success(f"Prediction: {result}")
