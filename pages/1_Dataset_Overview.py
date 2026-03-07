import streamlit as st
import pandas as pd
import os

st.title("📊 Dataset Overview")

dataset_path = "dataset"  # folder where images exist

if os.path.exists(dataset_path):
    classes = os.listdir(dataset_path)
    st.write("Classes:", classes)

    total = 0
    data = {}

    for c in classes:
        count = len(os.listdir(os.path.join(dataset_path, c)))
        data[c] = count
        total += count

    st.write("Total Images:", total)
    st.bar_chart(data)
else:
    st.warning("Dataset folder not found. Upload dataset or set correct path.")
