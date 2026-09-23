# Animal Classifier Specification

## Goal

Build an image classifier that predicts which animal is present in a photograph and returns a confidence score plus the scores for every discovered animal class.

## Model

- Use transfer learning with `torchvision` MobileNetV2 and ImageNet weights by default.
- Freeze `model.features` during normal training.
- Replace `model.classifier[1]` with a new `torch.nn.Linear` whose output size is the discovered number of classes.
- Keep `model.features` in evaluation mode while training so frozen batch-normalisation layers do not update.
- `--from_scratch` is an explicitly supported experiment: do not use pretrained weights and make every layer trainable.

## Images and transforms

- Discover classes from immediate folder names; never hard-code animal names or the number of animals.
- Resize each image to `224x224`.
- Apply ImageNet normalization with mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.
- The exact same transform is used for training and prediction.

## Data preparation

- Read images from `<raw_data>/<animal_name>/`.
- If `<raw_data>` contains one wrapper directory rather than animal directories, use that wrapper's contents.
- Verify every supported image can be opened by Pillow.
- Require at least two animal folders and at least ten images per animal.
- Split each class into 80% training and 20% test using fixed seed `42` by default.
- Copy files into `<train_out>/<animal_name>/` and `<test_out>/<animal_name>/`.

## Quality gate

`evaluate.py` exits with code `1` when test accuracy is below `min_accuracy` (default `0.70`); otherwise it exits successfully.

## Saved model format

The model directory is a folder containing:

- `model.pt`: PyTorch `state_dict`.
- `classes.json`: JSON list of discovered animal names in model class-index order.

## Acceptance criteria

- **AC1:** The Pillow/pytest data checks pass.
- **AC2:** `prep.py` finds five animals and creates train/test folders.
- **AC3:** `train.py` finishes and saves `model.pt` and `classes.json`.
- **AC4:** `evaluate.py` prints accuracy and a confusion matrix and writes `metrics.json`.
- **AC5:** Test accuracy is at least `0.70` and the quality gate passes.
- **AC6:** `predict.py` prints an animal and confidence for a demo image.
- **AC7:** `score.py` returns an animal and confidence for `sample-request.json`.
- **AC8:** No script hard-codes the number or names of animals.
- **AC9:** `README.md` explains every required step.