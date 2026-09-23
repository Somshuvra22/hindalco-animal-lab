"""Local Streamlit UI for the saved animal classifier."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


# Make the shared project model utilities importable when Streamlit starts
# this file from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_utils import load_model, predict_image  # noqa: E402


MODEL_DIR = PROJECT_ROOT / "models"

st.set_page_config(page_title="Animal Predictor")
st.title("Animal Predictor")
st.write("Upload a photo and click Predict to identify the animal locally.")
st.info("This version runs on the local saved model. Azure is not required.")

if not (MODEL_DIR / "model.pt").is_file() or not (MODEL_DIR / "classes.json").is_file():
    st.error(f"Local model files were not found in {MODEL_DIR}.")
    st.code(
        "python src/prep.py --raw_data data/animals "
        "--train_out split/train --test_out split/test\n"
        "python src/train.py --train_data split/train --model_dir models"
    )
    st.stop()

@st.cache_resource
def get_model():
    return load_model(MODEL_DIR)


uploaded_file = st.file_uploader("Choose an animal photo", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.info("Choose a JPG or PNG image above to begin.")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as error:
    st.error(f"The uploaded file could not be opened as an image: {error}")
    st.stop()

st.image(image, caption="Uploaded image", use_container_width=True)
st.success(f"Image loaded successfully ({image.width} × {image.height} pixels).")

if st.button("Predict", type="primary"):
    with st.spinner("Running the local model..."):
        try:
            model, classes = get_model()
            result = predict_image(model, classes, image)
        except Exception as error:
            st.error(f"Local prediction failed: {error}")
            st.stop()

    st.subheader(f"Prediction: {result['animal']}")
    st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
    scores = pd.DataFrame(
        {"animal": list(result["all_scores"]),
         "score": list(result["all_scores"].values())}
    ).set_index("animal")
    st.subheader("Scores for all animals")
    st.bar_chart(scores)
    if result["confidence"] < 0.60:
        st.warning("Confidence is below 60%; treat this prediction carefully.")