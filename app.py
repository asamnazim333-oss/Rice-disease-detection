import streamlit as st

st.set_page_config(page_title="🌾 Rice Disease App", layout="wide")

st.title("🌾 Rice Disease Detection App")
st.write("Use sidebar to navigate pages.")

st.sidebar.success("Select a page above.")
st.markdown("""
<style>
    .main {
        background-color: #f5f7ff;
    }
    h1 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)
