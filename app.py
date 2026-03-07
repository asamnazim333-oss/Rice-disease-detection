import streamlit as st

st.set_page_config(
    page_title="🌾 Rice Disease Detection",
    layout="wide"
)

st.title("🌾 Rice Disease Detection App")

st.markdown("""
Use the left sidebar to navigate between pages:
- Dataset Overview
- Preprocessing
- Classification
- Model Evaluation
""")
