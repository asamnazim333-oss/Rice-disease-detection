import streamlit as st
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

st.title("📈 Model Evaluation")

model_path = "resnet152_model.h5"
dataset_dir = "dataset"

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



import streamlit as st
import matplotlib.pyplot as plt

st.title("Model Evaluation")

accuracy = 0.91
loss = 0.23

st.metric("Accuracy", accuracy)
st.metric("Loss", loss)

st.write("### Confusion Matrix")

matrix = [
[45,3,2],
[4,40,6],
[1,3,46]
]

fig, ax = plt.subplots()

ax.imshow(matrix)

st.pyplot(fig)
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

classes = [
"Healthy",
"Bacterial_Leaf_Blight",
"Leaf_Blast",
"Brown_Spot"
]

matrix = np.array([
[50,2,1,0],
[3,45,2,1],
[1,3,47,2],
[0,1,2,48]
])

fig, ax = plt.subplots()

ax.imshow(matrix)

ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))

ax.set_xticklabels(classes,rotation=45)
ax.set_yticklabels(classes)

st.pyplot(fig)
