import streamlit as st
from groq import Groq
from PIL import Image
import base64
import io

st.title("🤖 Classification using Groq AI")

client = Groq(api_key="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")

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image")

    img64 = image_to_base64(image)

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Classify this rice disease image: data:image/png;base64,{img64}"
            }
        ],
        model="llama-3.2-11b-vision-preview"
    )

    st.success(response.choices[0].message.content)
