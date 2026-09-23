# Validation Report

Validation was run from `E:\vbc_p1` on Windows PowerShell with the project-local `E:\vbc_p1\.venv\Scripts\python.exe` environment and Python 3.12.5.

| Criterion | Result | Evidence |
|---|---|---|
| AC1 | PASS | `python -m pytest tests -v` reported `5 passed in 1.53s`. |
| AC2 | PASS | `prep.py` printed `cat`, `chicken`, `cow`, `dog`, and `horse`, each with `16` train and `4` test images, followed by `Found 5 animals`. |
| AC3 | PASS | Training printed `Saved model to E:\\vbc_p1\\models`; `models\\model.pt` exists (9,144,130 bytes) and `models\\classes.json` exists (59 bytes). |
| AC4 | PASS | Evaluation printed `Overall accuracy: 1.0000`, per-animal results, and `Confusion matrix (rows = real, columns = predicted)`; `metrics\\metrics.json` was written. |
| AC5 | PASS | Evaluation printed `QUALITY GATE PASSED: 1.0000 >= 0.7000`. |
| AC6 | PASS | `predict.py` printed `Prediction: cat  (confidence 84%)` for `demo_images/cat_demo.jpg`. |
| AC7 | PASS | `score.py` returned `{'animal': 'cat', 'confidence': 0.8482905030250549, 'all_scores': ...}` for `sample-request.json`. |
| AC8 | PASS | Source scripts discover classes from directory names or `ImageFolder`; no animal names or fixed class count are used in executable scripts. |
| AC9 | PASS | `README.md` documents installation, tests, preparation, training, evaluation, local prediction, scoring, and Streamlit usage. |

## Commands run

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r app\requirements.txt
.venv\Scripts\python.exe -m pytest tests -v
.venv\Scripts\python.exe src/prep.py --raw_data data/animals --train_out split/train --test_out split/test
.venv\Scripts\python.exe src/train.py --train_data split/train --model_dir models
.venv\Scripts\python.exe src/evaluate.py --model_dir models --test_data split/test --metrics_out metrics
.venv\Scripts\python.exe src/predict.py --model_dir models --image demo_images/cat_demo.jpg
.venv\Scripts\python.exe src/make_request.py --image demo_images/cat_demo.jpg --out sample-request.json
$env:AZUREML_MODEL_DIR = (Resolve-Path models).Path
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); import score; score.init(); print(score.run(open('sample-request.json', encoding='utf-8').read()))"
```

The generated `models/`, `split/`, `metrics/`, and `sample-request.json` artifacts are excluded by `.gitignore` as required.