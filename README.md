# Heart Disease Prediction

End-to-end machine-learning project that predicts the presence of heart disease from 13 clinical measurements, compares three models, explains the best one, and serves it through a Streamlit app.

> Educational portfolio project. Not a medical device and not for clinical use.

**Live demo:** *add your Streamlit Community Cloud link here*

## What it does

1. **Cleans** the UCI Heart Disease data: binarises the label, finds and handles missing values, treats impossible zeros as missing, removes duplicates.
2. **Preprocesses** inside a scikit-learn `Pipeline`: median / most-frequent imputation, scaling of numeric features, one-hot encoding of categorical features. Everything is fitted on training data only, so there is no leakage into the test set.
3. **Trains and compares** Logistic Regression (baseline), Random Forest and XGBoost, each tuned with 5-fold cross-validation.
4. **Evaluates** on a stratified 20% hold-out set: accuracy, precision, recall, F1, ROC-AUC and confusion matrices.
5. **Explains** the XGBoost model with SHAP.
6. **Serves** all three models in a Streamlit app.

## Results

Test-set results (stratified 80/20 split, `random\_state=42`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC (train) | TN | FP | FN | TP |

|---|---|---|---|---|---|---|---|---|---|---|

| Logistic Regression | 0.81 | 0.807 | 0.863 | 0.834 | 0.896 | 0.875 | 61 | 21 | 14 | 88 |

| Random Forest | 0.804 | 0.806 | 0.853 | 0.829 | 0.906 | 0.877 | 61 | 21 | 15 | 87 |

| XGBoost | 0.837 | 0.833 | 0.882 | 0.857 | 0.914 | 0.872 | 64 | 18 | 12 | 90 |

!\[Model comparison](results/figures/metrics\_comparison.png)
!\[Confusion matrices](results/figures/confusion\_matrices.png)
!\[ROC curves](results/figures/roc\_curves.png)
!\[Feature importance](results/figures/feature\_importance.png)
!\[SHAP summary](results/figures/shap\_summary.png)

## Data cleaning decisions

* **Label:** the original target has values 0 to 4. It is converted to binary: 0 = no disease, 1 to 4 = disease present.
* **Impossible zeros:** cholesterol and resting blood pressure recorded as `0` cannot be real measurements, so they are treated as missing.
* **Missing values:** imputed inside the pipeline (median for numeric, most frequent for categorical), so statistics come from training folds only.
* **Hospital column dropped:** the source hospital is not something a user can enter in the app and could let a model learn site effects instead of medicine.
* **Duplicates:** exact duplicate rows are dropped.

`python src/train.py` prints the exact counts for each step.

## Project structure

```
data/            dataset (see data/README.md)
src/preprocess.py  loading, cleaning, preprocessing pipeline
src/train.py       model training, comparison, figures, saved models
src/explain.py     SHAP explainability for XGBoost
app/streamlit\_app.py  web app
models/          trained pipelines + input schema (generated)
results/         metrics table and figures (generated)
tests/           unit tests for the cleaning and preprocessing code
```

## Run it

```bash
python -m venv .venv \&\& source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# put heart\_disease\_uci.csv in data/  (see data/README.md)
python src/train.py        # trains, evaluates, saves models and figures
python src/explain.py      # SHAP figures
pytest                     # unit tests
streamlit run app/streamlit\_app.py
```

## Deploy

Push the repo (including `models/` and `results/`) to GitHub, then create an app on [Streamlit Community Cloud](https://streamlit.io/cloud) pointing at `app/streamlit\_app.py`. If you retrain, commit the new `models/` folder so the app and the models stay in sync.

## Limitations

* Small dataset (about 900 patients) from four hospitals, collected decades ago.
* Several columns (`ca`, `thal`, `slope`) have a lot of missing values, and the missingness differs by hospital.
* A hold-out set of this size gives noisy metrics; differences of a few points between models are not conclusive.
* Predictions are associations in historical data, not clinical advice.

## Data source

Janosi, A., Steinbrunn, W., Pfisterer, M., Detrano, R. (1988). Heart Disease. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/Heart+Disease

