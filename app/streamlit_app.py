"""Streamlit app: heart disease risk estimator (educational demo, not medical advice)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
UNKNOWN = "Unknown"

st.set_page_config(page_title="Heart Disease Risk Estimator", page_icon="🫀", layout="wide")


@st.cache_resource
def load_artifacts():
    schema_path = MODELS_DIR / "schema.json"
    if not schema_path.exists():
        return None, None
    schema = json.loads(schema_path.read_text())
    models = {name: joblib.load(MODELS_DIR / file) for name, file in schema["models"].items()}
    return schema, models


def feature_input(meta):
    """Render one input widget from its schema entry and return the value (NaN if unknown)."""
    label, help_text = meta["label"], meta["description"]

    if meta["type"] == "numeric":
        lo, hi = meta["min"], meta["max"]
        if meta["integer"]:
            default = int(min(max(round(meta["median"]), lo), hi))
            return st.number_input(label, min_value=int(lo), max_value=int(hi), value=default,
                                   step=1, help=help_text)
        default = float(min(max(round(meta["median"], 1), lo), hi))
        return st.number_input(label, min_value=float(lo), max_value=float(hi), value=default,
                               step=0.1, help=help_text)

    options = list(meta["options"])
    if meta["nullable"]:
        options.append(UNKNOWN)
    choice = st.selectbox(label, options, index=options.index(meta["mode"]),
                          format_func=str, help=help_text)
    if choice == UNKNOWN:
        return np.nan
    return float(choice) if meta["type"] == "discrete" else choice


schema, models = load_artifacts()

st.title("🫀 Heart disease risk estimator")
st.caption(
    "Educational machine-learning demo trained on the UCI Heart Disease dataset. "
    "It is not a medical device and must not be used for diagnosis or treatment decisions."
)

if schema is None:
    st.error("No trained models found. Run `python src/train.py` first, then restart the app.")
    st.stop()

features = schema["features"]
tab_predict, tab_compare, tab_about = st.tabs(["Predict", "Model comparison", "About"])

with tab_predict:
    with st.form("patient_form"):
        model_name = st.selectbox("Model", list(models), index=list(models).index(schema["best_model"]))
        columns = st.columns(3)
        values = {}
        for i, (name, meta) in enumerate(features.items()):
            with columns[i % 3]:
                values[name] = feature_input(meta)
        submitted = st.form_submit_button("Estimate risk")

    if submitted:
        row = pd.DataFrame([values], columns=list(features))
        categorical = [n for n, m in features.items() if m["type"] == "categorical"]
        row[categorical] = row[categorical].astype(object)

        probabilities = {name: float(m.predict_proba(row)[0, 1]) for name, m in models.items()}
        prob = probabilities[model_name]
        band = "Lower" if prob < 0.35 else "Moderate" if prob < 0.65 else "Higher"

        left, right = st.columns([1, 2])
        left.metric(f"{model_name}: probability of heart disease", f"{prob:.0%}")
        right.progress(prob, text=f"{band} estimated risk")

        st.subheader("All three models on the same input")
        st.dataframe(
            pd.DataFrame({
                "Model": list(probabilities),
                "Probability of disease": [round(p, 3) for p in probabilities.values()],
                "Prediction": ["Disease" if p >= 0.5 else "No disease" for p in probabilities.values()],
            }),
            hide_index=True,
        )
        st.caption("Features marked 'Unknown' are filled in with the training-set median / most common value.")

with tab_compare:
    metrics_path = RESULTS_DIR / "metrics.csv"
    if metrics_path.exists():
        st.subheader("Held-out test set (20% of patients)")
        st.dataframe(pd.read_csv(metrics_path), hide_index=True)
    else:
        st.info("Run `python src/train.py` to generate the comparison.")
    for filename, caption in [
        ("metrics_comparison.png", "Accuracy, precision, recall, F1 and ROC-AUC"),
        ("confusion_matrices.png", "Confusion matrices"),
        ("roc_curves.png", "ROC curves"),
        ("feature_importance.png", "Most influential features per model"),
        ("shap_summary.png", "SHAP summary for XGBoost"),
    ]:
        path = FIGURES_DIR / filename
        if path.exists():
            st.image(str(path), caption=caption)

with tab_about:
    st.markdown(
        """
**Data.** UCI Heart Disease dataset (Cleveland, Hungary, Switzerland and VA Long Beach).
The label is binarised: 0 = no disease, 1 = disease present.

**Pipeline.** Impossible zeros in cholesterol / blood pressure are treated as missing, then
missing values are imputed (median / most frequent), numeric features are scaled and
categorical features are one-hot encoded, all inside a scikit-learn pipeline fitted on
training data only. Logistic Regression, Random Forest and XGBoost are tuned with 5-fold
cross-validation and compared on a held-out test set.

**Limitations.** Roughly 900 patients from four hospitals, collected decades ago, with
substantial missing data in some columns. Results do not generalise to current clinical
practice. This app is a portfolio demo.
"""
    )
