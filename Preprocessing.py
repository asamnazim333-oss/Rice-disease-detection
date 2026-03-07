import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

st.title("🧹 Preprocessing")

st.write("""
Preprocessing steps:
1. Resize image to 224x224
2. Convert to tensor
3. Normalize values
""")

uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Original Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    tensor = transform(image)
    st.write("Tensor Shape:", tensor.shape)
