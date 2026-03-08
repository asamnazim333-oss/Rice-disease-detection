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

    # Convert image to RGB (important)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # No resizing — keep full resolution
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
    probs = probs.tolist()

    labels = ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]

    # build dictionary of all probabilities
    pred_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}

    # best prediction
    best_label = max(pred_dict, key=pred_dict.get)
    best_conf = pred_dict[best_label]

    return best_label, best_conf, pred_dict
