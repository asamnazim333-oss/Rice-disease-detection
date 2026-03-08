import streamlit as st
import os
import matplotlib.pyplot as plt

st.title("🌾 Dataset Overview")

dataset_path = "Dataset"

if not os.path.exists(dataset_path):
    st.error("Dataset folder not found!")
    st.stop()

classes = os.listdir(dataset_path)

st.write("### Classes Found")

for c in classes:
    st.write("✔", c)

st.write("### Image Distribution")

counts = []

for c in classes:
    folder = os.path.join(dataset_path, c)
    if os.path.exists(folder):
        counts.append(len(os.listdir(folder)))
    else:
        counts.append(0)

fig, ax = plt.subplots()
ax.bar(classes, counts)
ax.set_xlabel("Class")
ax.set_ylabel("Number of Images")

st.pyplot(fig)

st.write("### Sample Images")

for c in classes:
    folder = os.path.join(dataset_path, c)

    st.write(f"#### {c}")

    if os.path.exists(folder):
        images = os.listdir(folder)[:4]  # show first 4 images

        cols = st.columns(4)

        for i, img in enumerate(images):
            img_path = os.path.join(folder, img)
            cols[i % 4].image(img_path, caption=img, use_container_width=True)
    else:
        st.write("No images found")
