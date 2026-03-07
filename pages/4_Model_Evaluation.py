import streamlit as st
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

st.title("📈 Model Evaluation")

model_path = "resnet152_model.h5"
dataset_dir = "Dataset"

if os.path.exists(model_path) and os.path.exists(dataset_dir):
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded!")

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    loss, acc = model.evaluate(val_gen)
    st.write(f"Validation Loss: {loss:.4f}")
    st.write(f"Validation Accuracy: {acc*100:.2f}%")

    # Optionally, show some predictions
    images, labels = next(val_gen)
    preds = model.predict(images)
    for i in range(5):
        st.image(images[i])
        st.write("Pred:", preds[i].argmax(), "Actual:", labels[i].argmax())

else:
    st.warning("Model or dataset not found!")
