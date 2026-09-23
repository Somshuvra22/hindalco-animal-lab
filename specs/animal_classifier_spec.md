# Animal Classifier Specification

## Goal

Build a beginner-friendly image classifier that predicts which animal appears
in a photo and returns both the predicted animal and a confidence score.

## Model

- Use torchvision MobileNetV2 transfer learning with ImageNet weights by
  default.
- Freeze `model.features` during transfer-learning training.
- Replace `model.classifier[1]` with a new `torch.nn.Linear` layer whose output
  size is discovered from the animal folders.
- Keep `model.features` in evaluation mode while the classifier is trained.
- `--from_scratch` is an explicit experiment: do not load pretrained weights
  and leave every model layer trainable.

## Image preprocessing

Every training, evaluation, and prediction image uses the same transform:
resize to 224x224, convert to a tensor, and apply ImageNet normalization
(mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]`).

## Classes and data

- Animal names are read from immediate subfolder names. No animal name or
  fixed class count is hard-coded.
- A future sixth animal works without code changes.
- Preparation validates every image can be opened, requires at least two
  animals and at least ten images per animal, and copies images into matching
  train/test class folders.
- Each animal is split 80% train and 20% test, using fixed seed 42 by default.
- If the raw data contains one wrapper folder, its contents are used.

## Model artifact

The saved model is a folder containing:

- `model.pt`: model `state_dict`.
- `classes.json`: JSON list of animal names in model class-index order.

Loading searches the supplied model directory and its subfolders for both
files, allowing Azure ML output-folder layouts.

## Quality gate

Evaluation writes `metrics.json`, prints overall and per-animal accuracy plus
a confusion matrix (rows are real labels and columns are predicted labels),
and exits with code 1 when test accuracy is below `min_accuracy` (default
`0.70`).

## Command-line interfaces

- `prep.py --raw_data --train_out --test_out [--test_ratio 0.2] [--min_images 10] [--seed 42]`
- `train.py --train_data --model_dir [--epochs 10] [--lr 0.001] [--batch_size 16] [--from_scratch]`
- `evaluate.py --model_dir --test_data --metrics_out [--min_accuracy 0.70]`
- `predict.py --model_dir --image`
- `make_request.py --image [--out sample-request.json]`

The Azure ML scoring interface loads from `AZUREML_MODEL_DIR` in `init()` and
accepts `{"image": "<base64 image>"}` in `run(raw_data)`.

## Acceptance criteria

- **AC1:** Pillow/pytest data checks pass.
- **AC2:** Preparation finds five animals and creates train/test folders.
- **AC3:** Training finishes and saves `model.pt` and `classes.json`.
- **AC4:** Evaluation prints accuracy and a confusion matrix and writes
  `metrics.json`.
- **AC5:** Test accuracy is at least 0.70 and the quality gate passes.
- **AC6:** Prediction prints an animal and confidence for a demo image.
- **AC7:** Scoring returns animal and confidence for a generated request.
- **AC8:** No script hard-codes animal names or the number of animals.
- **AC9:** README explains every step.