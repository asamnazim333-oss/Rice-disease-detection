import tensorflow as tf
import numpy as np

# Load model (ResNet50 or your trained model)
def load_model():
    model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=True
    )
    return model

def predict(model, img):
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)

    preds = model.predict(img)
    decoded = tf.keras.applications.resnet50.decode_predictions(preds, top=1)[0][0]

    label = decoded[1]
    confidence = float(decoded[2])

    return label, confidence
