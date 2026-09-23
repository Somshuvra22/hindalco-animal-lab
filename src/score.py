"""Azure ML online endpoint scoring script."""
import base64
import io
import json
import os
from PIL import Image
from model_utils import load_model, predict_image

_model = None
_classes = None


def init():
    global _model, _classes
    _model, _classes = load_model(os.environ["AZUREML_MODEL_DIR"])


def run(raw_data):
    try:
        body = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        image = Image.open(io.BytesIO(base64.b64decode(body["image"])))
        result = predict_image(_model, _classes, image)
        print(f"Prediction: {result['animal']} confidence={result['confidence']:.4f}")
        return result
    except Exception as error:
        return {"error": str(error)}