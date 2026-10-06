"""Explain the trained XGBoost model with SHAP (global feature impact).

Usage (after running train.py):
    python src/explain.py --data data/heart_disease_uci.csv

Writes results/figures/shap_summary.png and shap_importance.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from preprocess import clean_feature_names, load_and_clean, split_data

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "data" / "heart_disease_uci.csv"))
    args = parser.parse_args()

    df, _ = load_and_clean(args.data)
    _, X_test, _, _ = split_data(df)  # same split as train.py

    pipe = joblib.load(ROOT / "models" / "xgboost.joblib")
    preprocess, model = pipe.named_steps["preprocess"], pipe.named_steps["model"]
    X_t = pd.DataFrame(preprocess.transform(X_test), columns=clean_feature_names(preprocess))

    explanation = shap.TreeExplainer(model)(X_t)
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    shap.plots.beeswarm(explanation, max_display=12, show=False)
    plt.title("SHAP summary: how each feature pushes the XGBoost prediction")
    plt.savefig(fig_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    shap.plots.bar(explanation, max_display=12, show=False)
    plt.title("Mean |SHAP value| per feature")
    plt.savefig(fig_dir / "shap_importance.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP figures to {fig_dir}")


if __name__ == "__main__":
    main()
