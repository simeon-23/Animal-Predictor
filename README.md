# Animal Predictor

This project classifies animal photographs using a torchvision MobileNetV2 model. Classes are discovered from folder names, so adding another animal folder requires no code change.

## 1. Install dependencies

From `E:\vbc_p1` in PowerShell, use the project-local `.venv` environment for every command:

```powershell
if (-not (Test-Path .venv\Scripts\python.exe)) { python -m venv .venv }
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r app\requirements.txt
```

## 2. Check the source data

```powershell
.venv\Scripts\python.exe -m pytest tests -v
```

The source data must contain at least two animal folders, at least ten readable images per folder, and no loose images at the top level.

## 3. Prepare the split

```powershell
.venv\Scripts\python.exe src/prep.py --raw_data data/animals --train_out split/train --test_out split/test
```

This creates an 80/20 per-animal split using seed 42.

## 4. Train

```powershell
.venv\Scripts\python.exe src/train.py --train_data split/train --model_dir models
```

The default uses ImageNet-pretrained MobileNetV2, freezes its feature extractor, and trains the replacement classifier. To run the optional experiment from scratch:

```powershell
.venv\Scripts\python.exe src/train.py --train_data split/train --model_dir models-scratch --from_scratch
```

The saved model folder contains `model.pt` and `classes.json`.

## 5. Evaluate

```powershell
.venv\Scripts\python.exe src/evaluate.py --model_dir models --test_data split/test --metrics_out metrics
```

Evaluation prints overall accuracy, per-animal accuracy, and a confusion matrix. It writes `metrics/metrics.json` and exits with code 1 when accuracy is below 0.70.

## 6. Predict a local image

```powershell
.venv\Scripts\python.exe src/predict.py --model_dir models --image demo_images/cat_demo.jpg
```

This prints the prediction and saves an annotated image with the animal and confidence text to `outputs/prediction.jpg`. Use `--output another-name.jpg` to choose a different output path.

## 7. Test the Azure scoring script locally

```powershell
.venv\Scripts\python.exe src/make_request.py --image demo_images/cat_demo.jpg
$env:AZUREML_MODEL_DIR = (Resolve-Path models).Path
.venv\Scripts\python.exe -c "import json, sys; sys.path.insert(0, 'src'); import score; score.init(); print(score.run(open('sample-request.json', encoding='utf-8').read()))"
```

The request creator shrinks images to at most 512 pixels and encodes a JPEG to keep endpoint requests small.

## 8. Run the Streamlit app

Install the UI dependencies and set the endpoint values in PowerShell:

```powershell
$env:ENDPOINT_URL = 'https://your-endpoint-url'
$env:ENDPOINT_KEY = 'your-endpoint-key'
.venv\Scripts\streamlit.exe run app/app.py
```

The app uploads a compressed image, sends the required bearer token, displays the prediction, confidence, all-score bar chart, and a warning below 60% confidence.

## Azure ML files

The provided files under `azureml/` and `azure-pipelines.yml` invoke the scripts in `src/` with the same arguments documented above. Configure the placeholders in those provided deployment files for your own Azure ML workspace; this project uses no other cloud services.