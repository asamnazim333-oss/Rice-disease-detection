from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import numpy as np

model_name = "prithivMLmods/Rice-Leaf-Disease"

def load_model():
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor

def predict(model_data, img):
    model, processor = model_data

    # Convert image to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Preprocess
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
    probs = probs.tolist()

    labels = ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]

    best_idx = int(np.argmax(probs))
    label = labels[best_idx]
    confidence = float(probs[best_idx])

    return label, confidence
