import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

st.title("🤖 Classification")

# Load pretrained model (cached)
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Example classes (you can replace with rice disease labels)
classes = [
    "Healthy",
    "Disease Type 1",
    "Disease Type 2",
    "Disease Type 3"
]

def predict(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    return classes[pred % len(classes)]

# Upload image
uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    result = predict(image)
    st.success(f"Prediction: {result}")
