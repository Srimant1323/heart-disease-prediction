"""Train and compare Logistic Regression, Random Forest and XGBoost.

Usage:
    python src/train.py --data data/heart_disease_uci.csv

Outputs (relative to the repo root):
    models/*.joblib, models/schema.json     trained pipelines + app input schema
    results/metrics.csv, results/metrics.md test-set comparison table
    results/best_params.json                hyperparameters chosen by cross-validation
    results/figures/*.png                   EDA, confusion matrices, ROC, metrics, importances
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocess import (
    CATEGORICAL,
    DESCRIPTIONS,
    DISCRETE,
    FEATURES,
    INTEGER_COLUMNS,
    LABELS,
    SEED,
    build_preprocessor,
    clean_feature_names,
    load_and_clean,
    split_data,
)

ROOT = Path(__file__).resolve().parent.parent

# Chart styling: first three slots of a colour-blind-checked categorical palette.
COLORS = {"Logistic Regression": "#2a78d6", "Random Forest": "#eb6834", "XGBoost": "#1baf7a"}
INK, INK2, MUTED, GRID, SURFACE, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def model_specs():
    """(estimator, hyperparameter grid) for each model. Grids are small on purpose."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 3, 5],
            },
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.03, 0.1],
            },
        ),
    }


def clean_axes(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def to_markdown(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in table.astype(object).itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def print_report(report: dict) -> None:
    print("\n=== Data cleaning report ===")
    print(f"Rows in file:                 {report['rows_raw']}")
    print(f"Rows without a label (dropped): {report['rows_without_label']}")
    print(f"Zeros treated as missing:     {report['zeros_set_to_missing']}")
    print(f"Duplicate rows dropped:       {report['duplicates_dropped']}")
    print(f"Rows used:                    {report['rows_clean']}")
    print(f"Class balance (0 = no disease, 1 = disease): {report['class_balance']}")
    print("Missing values per column:")
    for col, n in report["missing_per_column"].items():
        if n:
            print(f"  {col:9s} {n:4d}  ({n / report['rows_clean']:.0%})")


def save_eda(df: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})
    missing = (df[FEATURES].isna().mean() * 100).sort_values()
    ax1.barh(missing.index, missing.values, color=COLORS["Logistic Regression"], edgecolor=SURFACE, linewidth=1.5)
    ax1.set_title("Missing values by feature (before imputation)")
    ax1.set_xlabel("% of rows")
    clean_axes(ax1, "x")
    ax1.grid(axis="y", visible=False)

    counts = df["target"].value_counts().sort_index()
    ax2.bar(["No disease", "Disease"], counts.values,
            color=[COLORS["Logistic Regression"], COLORS["Random Forest"]], edgecolor=SURFACE, linewidth=1.5)
    for x, v in enumerate(counts.values):
        ax2.text(x, v, str(v), ha="center", va="bottom", fontsize=10, color=INK2)
    ax2.set_title("Class balance")
    ax2.set_ylabel("Patients")
    clean_axes(ax2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confusion_matrices(y_test, preds, path: Path) -> None:
    fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 4))
    for ax, (name, y_pred) in zip(axes, preds.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax, display_labels=["No disease", "Disease"],
            cmap="Blues", colorbar=False, values_format="d",
        )
        ax.set_title(name)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_roc(y_test, probas, aucs, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, prob in probas.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{name} (AUC {aucs[name]:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(frameon=False, loc="lower right")
    clean_axes(ax, "both")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_metric_bars(table: pd.DataFrame, path: Path) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (_, row) in enumerate(table.iterrows()):
        name = row["Model"]
        values = [float(row[m]) for m in metrics]
        xs = [j + (i - 1) * width for j in range(len(metrics))]
        ax.bar(xs, values, width=width, color=COLORS[name], edgecolor=SURFACE, linewidth=1.5, label=name)
        for x, v in zip(xs, values):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model comparison on the held-out test set")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_importances(fitted, path: Path, top: int = 10) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (name, pipe) in zip(axes, fitted.items()):
        names = clean_feature_names(pipe.named_steps["preprocess"])
        model = pipe.named_steps["model"]
        if name == "Logistic Regression":
            values, xlabel = abs(model.coef_[0]), "|coefficient| (standardised inputs)"
        else:
            values, xlabel = model.feature_importances_, "Feature importance"
        series = pd.Series(values, index=names).sort_values().tail(top)
        ax.barh(series.index, series.values, color=COLORS[name], edgecolor=SURFACE, linewidth=1.5)
        ax.set_title(name)
        ax.set_xlabel(xlabel)
        clean_axes(ax, "x")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_schema(df: pd.DataFrame, model_files: dict, best_name: str) -> dict:
    """Describe the inputs so the Streamlit app can build its form from the data."""
    schema = {
        "best_model": best_name,
        "models": model_files,
        "target": "1 = heart disease present (original label > 0), 0 = absent",
        "features": {},
    }
    for col in FEATURES:
        series = df[col]
        meta = {
            "label": LABELS[col],
            "description": DESCRIPTIONS[col],
            "nullable": bool(series.isna().any()),
        }
        if col in CATEGORICAL:
            meta.update(type="categorical", options=sorted(series.dropna().unique().tolist()),
                        mode=str(series.mode().iloc[0]))
        elif col in DISCRETE:
            meta.update(type="discrete", options=sorted(int(v) for v in series.dropna().unique()),
                        mode=int(series.mode().iloc[0]))
        else:
            meta.update(type="numeric", min=float(series.min()), max=float(series.max()),
                        median=float(series.median()), integer=col in INTEGER_COLUMNS)
        schema["features"][col] = meta
    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "data" / "heart_disease_uci.csv"))
    args = parser.parse_args()

    models_dir, results_dir = ROOT / "models", ROOT / "results"
    fig_dir = results_dir / "figures"
    for folder in (models_dir, fig_dir):
        folder.mkdir(parents=True, exist_ok=True)

    # 1. load + clean
    df, report = load_and_clean(args.data)
    print_report(report)
    save_eda(df, fig_dir / "eda_overview.png")

    # 2. split, then train each model with 5-fold CV inside the training set only
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"\nTrain rows: {len(X_train)} | Test rows: {len(X_test)}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    fitted, preds, probas, aucs, rows, best_params = {}, {}, {}, {}, [], {}
    for name, (estimator, grid) in model_specs().items():
        print(f"Training {name} ...")
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        aucs[name] = roc_auc_score(y_test, y_prob)
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred), 3),
            "Recall": round(recall_score(y_test, y_pred), 3),
            "F1": round(f1_score(y_test, y_pred), 3),
            "ROC-AUC": round(aucs[name], 3),
            "CV ROC-AUC (train)": round(search.best_score_, 3),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        })
        fitted[name], preds[name], probas[name] = best, y_pred, y_prob
        best_params[name] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        joblib.dump(best, models_dir / f"{slug(name)}.joblib")

    table = pd.DataFrame(rows)
    # Choose the "default" model for the app using training-set CV only, never the test set.
    best_name = table.loc[table["CV ROC-AUC (train)"].idxmax(), "Model"]

    # 3. save tables, schema and figures
    table.to_csv(results_dir / "metrics.csv", index=False)
    (results_dir / "metrics.md").write_text(to_markdown(table) + "\n")
    (results_dir / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))
    model_files = {name: f"{slug(name)}.joblib" for name in fitted}
    (models_dir / "schema.json").write_text(json.dumps(build_schema(df, model_files, best_name), indent=2))

    save_confusion_matrices(y_test, preds, fig_dir / "confusion_matrices.png")
    save_roc(y_test, probas, aucs, fig_dir / "roc_curves.png")
    save_metric_bars(table, fig_dir / "metrics_comparison.png")
    save_importances(fitted, fig_dir / "feature_importance.png")

    print("\n=== Test-set results ===")
    print(table.to_string(index=False))
    print(f"\nBest model by cross-validated ROC-AUC (training data): {best_name}")
    print(f"Saved models to {models_dir} and results to {results_dir}")


if __name__ == "__main__":
    main()
