"""Shared model, image transform, prediction, and optional metric helpers."""
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


def get_transform():
    """Return the one transform used by training and prediction."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(num_classes, pretrained=True, freeze=True):
    """Build MobileNetV2 and adapt its final layer to the discovered classes."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    if freeze:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        model.features.eval()
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
    return model


def save_model(model, classes, model_dir):
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "classes.json").write_text(json.dumps(list(classes), indent=2), encoding="utf-8")


def load_model(model_dir):
    """Find an artifact pair in model_dir or any nested output folder."""
    root = Path(model_dir)
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    for folder in candidates:
        if (folder / "model.pt").is_file() and (folder / "classes.json").is_file():
            classes = json.loads((folder / "classes.json").read_text(encoding="utf-8"))
            model = build_model(len(classes), pretrained=False, freeze=False)
            state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            return model, classes
    raise FileNotFoundError(f"Could not find model.pt and classes.json below {model_dir}")


def predict_image(model, classes, pil_image):
    image = pil_image.convert("RGB")
    with torch.no_grad():
        scores = torch.softmax(model(get_transform()(image).unsqueeze(0)), dim=1)[0]
    index = int(torch.argmax(scores))
    return {
        "animal": classes[index],
        "confidence": float(scores[index]),
        "all_scores": {name: float(scores[i]) for i, name in enumerate(classes)},
    }


def log_metric(name, value, step=None):
    """Log only to Azure ML's MLflow endpoint; remain a no-op locally."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if uri.startswith("azureml"):
        import mlflow
        if step is None:
            mlflow.log_metric(name, value)
        else:
            mlflow.log_metric(name, value, step=step)