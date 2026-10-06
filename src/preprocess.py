"""Data loading, cleaning and preprocessing for the UCI Heart Disease dataset.

Works with the Kaggle mirror (``heart_disease_uci.csv``, 920 rows from four
hospitals) and with UCI-style files that use numeric codes (e.g. the Cleveland
subset), which are decoded into readable categories.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "target"

# Natural dataset order (also the order of the inputs in the app).
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalch", "exang", "oldpeak", "slope", "ca", "thal",
]
NUMERIC = ["age", "trestbps", "chol", "thalch", "oldpeak"]
DISCRETE = ["ca"]  # number of major vessels (0-3): numeric, but only a few values
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
INTEGER_COLUMNS = {"age", "trestbps", "chol", "thalch"}

# Column names differ slightly between sources.
RENAME = {"thalach": "thalch", "target": "num", "condition": "num"}

# Decoding of the numeric codes used in the raw UCI files.
CODE_MAPS = {
    "sex": {1: "Male", 0: "Female"},
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal", 4: "asymptomatic"},
    "fbs": {1: "True", 0: "False"},
    "restecg": {0: "normal", 1: "st-t abnormality", 2: "lv hypertrophy"},
    "exang": {1: "True", 0: "False"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversable defect"},
}

LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalch": "Maximum heart rate achieved (bpm)",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak-exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thalassemia (stress test result)",
}

DESCRIPTIONS = {
    "age": "Patient age in years.",
    "sex": "Sex as recorded in the dataset.",
    "cp": "Type of chest pain: typical angina, atypical angina, non-anginal pain or asymptomatic.",
    "trestbps": "Resting blood pressure on admission.",
    "chol": "Serum cholesterol.",
    "fbs": "Whether fasting blood sugar exceeded 120 mg/dl.",
    "restecg": "Resting electrocardiogram result.",
    "thalch": "Maximum heart rate achieved during the stress test.",
    "exang": "Whether exercise triggered angina.",
    "oldpeak": "ST depression induced by exercise relative to rest.",
    "slope": "Slope of the peak-exercise ST segment.",
    "ca": "Number of major vessels (0-3) coloured by fluoroscopy. Often not recorded.",
    "thal": "Result of the thallium stress test. Often not recorded.",
}


def _normalise_category(value):
    """Return a clean string label (or NaN) for one categorical cell."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return text.capitalize()
    return text


def load_and_clean(path):
    """Load the CSV and return ``(clean_dataframe, report)``.

    Cleaning steps:
      1. keep the 13 clinical features and binarise the label (0 = no disease,
         1-4 = disease present);
      2. decode numeric codes and tidy category labels;
      3. treat impossible zeros in cholesterol / blood pressure as missing;
      4. drop exact duplicate rows.

    Missing values are *not* filled here. Imputation happens inside the model
    pipeline so that it is learned from training data only (no leakage).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for download steps."
        )

    raw = pd.read_csv(path)
    raw.columns = [RENAME.get(c.strip().lower(), c.strip().lower()) for c in raw.columns]
    missing_cols = [c for c in FEATURES + ["num"] if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    df = raw[FEATURES + ["num"]].copy()
    report = {"rows_raw": int(len(df))}

    # 1. label
    df["num"] = pd.to_numeric(df["num"], errors="coerce")
    report["rows_without_label"] = int(df["num"].isna().sum())
    df = df.dropna(subset=["num"])
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")

    # 2. categorical columns
    for col in CATEGORICAL:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            # A True/False column with no blanks is read as bool by pandas.
            series = series.astype(object).map({True: "True", False: "False"})
        elif pd.api.types.is_numeric_dtype(series) and col in CODE_MAPS:
            # Numeric codes (possibly floats such as 1.0 when blanks are present).
            series = series.map(lambda v: CODE_MAPS[col].get(int(v)) if pd.notna(v) else np.nan)
        df[col] = series.map(_normalise_category).astype(object)

    for col in NUMERIC + DISCRETE:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. impossible zeros (the hospitals that did not measure these wrote 0)
    zeros = {}
    for col in ("chol", "trestbps"):
        mask = df[col] == 0
        zeros[col] = int(mask.sum())
        df.loc[mask, col] = np.nan
    report["zeros_set_to_missing"] = zeros

    # 4. duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_dropped"] = int(before - len(df))

    report["rows_clean"] = int(len(df))
    report["missing_per_column"] = {c: int(df[c].isna().sum()) for c in FEATURES}
    report["class_balance"] = {int(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()}
    return df, report


def split_data(df, test_size=0.2):
    """Stratified train/test split with a fixed seed."""
    return train_test_split(
        df[FEATURES], df[TARGET], test_size=test_size, stratify=df[TARGET], random_state=SEED
    )


def build_preprocessor():
    """Impute -> scale numeric columns, impute -> one-hot encode categorical ones."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, NUMERIC + DISCRETE), ("cat", categorical, CATEGORICAL)],
        sparse_threshold=0.0,
    )


def clean_feature_names(preprocessor):
    """Feature names after preprocessing, without the 'num__' / 'cat__' prefixes."""
    return [name.split("__", 1)[1] for name in preprocessor.get_feature_names_out()]
