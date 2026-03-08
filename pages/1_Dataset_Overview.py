import streamlit as st
import os
import matplotlib.pyplot as plt

st.title("Rice Disease Dataset Overview")

dataset_path = "dataset"

classes = os.listdir(dataset_path)

counts = []

for c in classes:
    path = os.path.join(dataset_path, c)
    counts.append(len(os.listdir(path)))

st.write("### Dataset Classes")

for c in classes:
    st.write(c)

st.write("### Image Distribution")

fig, ax = plt.subplots()

ax.bar(classes, counts)

ax.set_xlabel("Class")
ax.set_ylabel("Number of Images")

st.pyplot(fig)
st.write("### Sample Images")

for c in classes:
    folder = os.path.join(dataset_path, c)
    image = os.listdir(folder)[0]
    
    st.image(os.path.join(folder, image), caption=c, width=200)
