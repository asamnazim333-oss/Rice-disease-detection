from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import numpy as np
from PIL import Image

# Load the pretrained Hugging Face rice model
def load_model():
    model_name = "prithivMLmods/Rice-Leaf-Disease"
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor

def predict(model_data, img):
    model, processor = model_data
    
    # Ensure image is RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Compute probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()
    
    # Class labels according to the Hugging Face model
    labels = ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]
    
    # Find best prediction
    best_idx = int(np.argmax(probs))
    label = labels[best_idx]
    confidence = float(probs[best_idx])
    
    return label, confidence
