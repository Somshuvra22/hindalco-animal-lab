# Validation Report

Validation was run on Linux from the project root with the commands required
by the specification.

| Criterion | Result | Evidence |
|---|---|---|
| AC1 | PASS | `1 passed in 0.61s` from `python -m pytest tests -v`. |
| AC2 | PASS | `prep.py` printed five classes: cat, chicken, cow, dog, horse; each had `16` train and `4` test images. |
| AC3 | PASS | Training completed all 10 epochs and printed `Saved model to .../models`; `models/model.pt` and `models/classes.json` exist. |
| AC4 | PASS | Evaluation printed `Overall accuracy: 0.9000`, per-animal accuracy, and the rows-real/columns-predicted confusion matrix; `metrics/metrics.json` exists. |
| AC5 | PASS | Evaluation printed `QUALITY GATE PASSED: 0.9000 >= 0.7000`; `metrics.json` records accuracy `0.9`. |
| AC6 | PASS | Prediction printed `Prediction: cat  (confidence 86%)` for `demo_images/cat_demo.jpg`. |
| AC7 | PASS | `score.py` printed `Prediction: cat confidence=0.8558` and returned JSON containing `animal`, `confidence`, and `all_scores`. |
| AC8 | PASS | Static search found no animal-name or fixed-five-class literals in `src`; classes are read from folder names / `classes.json`. |
| AC9 | PASS | `README.md` documents installation, tests, preparation, training, evaluation, prediction, request generation, scoring, Azure usage, and Streamlit usage. |

Additional checks:

- `python -m compileall -q src app tests` completed successfully.
- The supplied `azureml/` and `azure-pipelines.yml` files were not edited.
- The generated request was a 15,245-byte JPEG payload, below the endpoint
  size limit described in the specification.