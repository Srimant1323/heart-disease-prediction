import numpy as np
import pandas as pd

from preprocess import CATEGORICAL, FEATURES, TARGET, build_preprocessor, load_and_clean


def _kaggle_style_frame():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 5],
        "age": [54, 61, 47, 70, 39, 39],
        "sex": ["Male", "Female", "Male", "Male", "Female", "Female"],
        "dataset": ["Cleveland"] * 6,
        "cp": ["asymptomatic", "typical angina", "non-anginal", "asymptomatic", "atypical angina", "atypical angina"],
        "trestbps": [130, 0, 120, 150, 118, 118],
        "chol": [250, 0, 210, 300, 190, 190],
        "fbs": [True, False, False, True, False, False],
        "restecg": ["normal", "normal", "lv hypertrophy", "normal", np.nan, np.nan],
        "thalch": [150, 140, 170, 120, 165, 165],
        "exang": [False, True, False, True, False, False],
        "oldpeak": [1.0, 2.3, 0.0, 3.1, 0.2, 0.2],
        "slope": ["flat", "flat", "upsloping", np.nan, "upsloping", "upsloping"],
        "ca": [0, 2, np.nan, 3, 0, 0],
        "thal": ["normal", "reversable defect", np.nan, "fixed defect", "normal", "normal"],
        "num": [0, 3, 0, 4, 0, 0],
    })


def _load(tmp_path, frame):
    path = tmp_path / "heart.csv"
    frame.to_csv(path, index=False)
    return load_and_clean(path)


def test_target_is_binary(tmp_path):
    df, _ = _load(tmp_path, _kaggle_style_frame())
    assert set(df[TARGET].unique()) == {0, 1}
    assert df[TARGET].sum() == 2  # num 3 and 4 -> disease


def test_impossible_zeros_become_missing(tmp_path):
    df, report = _load(tmp_path, _kaggle_style_frame())
    assert report["zeros_set_to_missing"] == {"chol": 1, "trestbps": 1}
    assert (df["chol"].dropna() > 0).all()


def test_duplicates_are_dropped(tmp_path):
    _, report = _load(tmp_path, _kaggle_style_frame())
    assert report["duplicates_dropped"] == 1


def test_uci_numeric_codes_are_decoded(tmp_path):
    frame = pd.DataFrame({
        "age": [60, 45], "sex": [1, 0], "cp": [4, 1], "trestbps": [140, 120], "chol": [240, 200],
        "fbs": [1, 0], "restecg": [2, 0], "thalach": [130, 160], "exang": [1, 0],
        "oldpeak": [2.0, 0.0], "slope": [2, 1], "ca": [1, 0], "thal": [7, 3], "num": [2, 0],
    })
    df, _ = _load(tmp_path, frame)
    assert df.loc[0, "sex"] == "Male"
    assert df.loc[0, "cp"] == "asymptomatic"
    assert df.loc[1, "thal"] == "normal"


def test_boolean_columns_without_blanks_are_kept(tmp_path):
    # pandas reads a True/False column with no blanks as bool; it must not turn into all-NaN.
    frame = _kaggle_style_frame().iloc[:5]
    df, _ = _load(tmp_path, frame)
    assert df["fbs"].isna().sum() == 0
    assert set(df["fbs"].unique()) == {"True", "False"}
    assert set(df["exang"].unique()) == {"True", "False"}


def test_numeric_codes_with_blanks_are_decoded(tmp_path):
    # Blanks make pandas read code columns as float (1.0, 0.0); they must still decode.
    frame = pd.DataFrame({
        "age": [60, 45, 52], "sex": [1, 0, 1], "cp": [4, 1, 3], "trestbps": [140, 120, 130],
        "chol": [240, 200, 220], "fbs": [1.0, np.nan, 0.0], "restecg": [2, 0, 1],
        "thalach": [130, 160, 150], "exang": [1.0, 0.0, np.nan], "oldpeak": [2.0, 0.0, 1.0],
        "slope": [2.0, np.nan, 1.0], "ca": [1, 0, 2], "thal": [7.0, 3.0, np.nan], "num": [2, 0, 0],
    })
    df, _ = _load(tmp_path, frame)
    assert df["fbs"].tolist()[0] == "True" and pd.isna(df["fbs"].tolist()[1])
    assert df["thal"].tolist()[0] == "reversable defect"
    assert df["slope"].tolist()[2] == "upsloping"


def test_preprocessor_handles_missing_inputs(tmp_path):
    df, _ = _load(tmp_path, _kaggle_style_frame())
    pre = build_preprocessor().fit(df[FEATURES])
    row = df[FEATURES].iloc[[0]].copy()
    row[["chol", "ca"]] = np.nan
    row[CATEGORICAL] = row[CATEGORICAL].astype(object)
    row.loc[:, "thal"] = np.nan
    out = pre.transform(row)
    assert not np.isnan(out).any()
    assert not hasattr(out, "toarray")  # dense output
