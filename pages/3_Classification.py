import streamlit as st
from openai import OpenAI
import os
from PIL import Image
import base64
import io

st.title("🤖 Rice Disease Classification (Groq AI)")

# Groq client (OpenAI style)
client = OpenAI(
    api_key=os.environ.get("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"),
    base_url="https://api.groq.com/openai/v1",
)

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image")

    img64 = image_to_base64(image)

    response = client.responses.create(
        input=f"Classify this rice disease image. Image base64: {img64}",
        model="openai/gpt-oss-20b"
    )

    st.success(response.output_text)
