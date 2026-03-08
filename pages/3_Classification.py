import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Image Augmentation Preview")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    img = tf.image.resize(img,(224,224))

    flip = tf.image.flip_left_right(img)
    rotate = tf.image.rot90(img)
    bright = tf.image.adjust_brightness(img,0.3)

    col1,col2,col3,col4 = st.columns(4)

    col1.image(img, caption="Original")
    col2.image(flip, caption="Flip")
    col3.image(rotate, caption="Rotate")
    col4.image(bright, caption="Brightness")
