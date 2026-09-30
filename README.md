# Mobile Price Range Prediction — MLOps Project

This project follows the same Lab 1–5 style structure as the supplied `churn-prediction-final` project, adapted to the uploaded `MLOP1` mobile-price classification notebook.

## Important

The uploaded notebook expects a dataset named `MLOP.CSV`. That CSV was not included with the uploaded files, so it is **not fabricated or replaced** here. Put your original `MLOP.CSV` into `data/raw/` before running the pipelines.

## Structure

```text
MLOP1-structured/
├── data/
│   ├── raw/MLOP.CSV                         # Add your original dataset here
│   └── processed/                           # Generated preprocessing outputs
├── notebooks/project_implementation.ipynb   # Uploaded MLOP1 notebook
├── src/
│   ├── preprocess.py                        # Lab 1 preprocessing
│   ├── train.py                             # Lab 2/3 baseline training
│   ├── evaluate.py                          # Lab 2/3 evaluation
│   ├── train_mlflow.py                      # Lab 4 MLflow tracking
│   ├── validate_reproducibility.py          # Lab 4 reproducibility check
│   ├── validate_data.py                     # Lab 5 schema/data validation
│   ├── preprocess_pipeline.py               # Lab 5 preprocessing pipeline
│   └── validate_outputs.py                  # Lab 5 output validation
├── pipelines/
│   ├── run_lab3_baseline.py
│   ├── run_lab4_tracking.py
│   └── run_lab5_pipeline.py
├── models/                                  # Generated model/preprocessor files
├── outputs/                                 # Generated evaluation/error outputs
├── artifacts/                               # Generated reports and plots
├── logs/
├── mlruns/                                  # MLflow tracking store
├── requirements.txt
└── .gitignore
```

## Notebook facts carried into this structure

- Dataset: 2,000 observations and 21 columns.
- Target: `price_range` with four classes (0, 1, 2, 3).
- Split: 80% training / 20% testing, stratified, `random_state=42`.
- Input features: 20 numerical features.
- The notebook compares Logistic Regression, KNN, Decision Tree, Random Forest and SVM.
- The notebook's tuned Random Forest uses GridSearchCV with 5-fold CV.
- Reported tuned-model test accuracy: 0.8825; weighted precision: 0.882436; weighted recall: 0.8825; weighted F1: 0.882415.

## Getting started on macOS/Linux

```bash
cd MLOP1-structured
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place your original dataset here:

```text
data/raw/MLOP.CSV
```

Then run:

```bash
python src/preprocess.py
python pipelines/run_lab3_baseline.py
python pipelines/run_lab4_tracking.py
python pipelines/run_lab5_pipeline.py
```

The scripts are designed to use the same 20-feature `price_range` dataset described by the uploaded notebook.


## Lab 6 — MLflow Model Registry

The Lab 6 registry script registers the latest completed model run from the
`Mobile_Price_Range_Prediction` MLflow experiment as
`Mobile_Price_Range_Model` and assigns the `candidate` alias to the new version.

Run Lab 4 first so that at least one completed MLflow model run exists:

```bash
python pipelines/run_lab4_tracking.py
python pipelines/run_lab6_registry.py
mlflow ui
```

The registry uses MLflow model aliases rather than deprecated model stages.
The `candidate` alias identifies the version currently selected for review.
