# Animal Classifier

This project trains a MobileNetV2 image classifier from animal-named folders.
The class list is discovered from the folders, so adding another class needs no
code change.

## Local setup and workflow

On Linux, from the project root:

```bash
python -m pip install -r requirements.txt
python -m pytest tests -v
python src/prep.py --raw_data data/animals --train_out split/train --test_out split/test
python src/train.py --train_data split/train --model_dir models
python src/evaluate.py --model_dir models --test_data split/test --metrics_out metrics
python src/predict.py --model_dir models --image demo_images/cat_demo.jpg
python src/make_request.py --image demo_images/cat_demo.jpg
```

Evaluation prints accuracy and exits with status 1 if accuracy is below 0.70.
The model artifact contains `models/model.pt` and `models/classes.json`.

Use `--epochs`, `--lr`, and `--batch_size` to tune training. Add
`--from_scratch` to train all layers without ImageNet weights.

## Scoring test

```bash
AZUREML_MODEL_DIR=models python - <<'PY'
import json, os, sys
sys.path.insert(0, "src")
import score
score.init()
print(score.run(open("sample-request.json").read()))
PY
```

Azure ML uses `azureml/pipeline.yml` and deploys `src/score.py`. Do not change
the supplied Azure configuration. The scoring request is JSON with one
base64-encoded JPEG field named `image`.

## Streamlit app

Install the UI dependencies and configure the endpoint:

```bash
python -m pip install -r app/requirements.txt
export ENDPOINT_URL="https://..."
export ENDPOINT_KEY="..."
streamlit run app/app.py
```

The UI shrinks uploads to at most 512 pixels, sends the required bearer token,
and displays the prediction, confidence, score chart, and low-confidence
warning.