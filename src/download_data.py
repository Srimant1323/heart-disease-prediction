"""Optional helper: download the Cleveland subset of the UCI Heart Disease data.

    pip install ucimlrepo
    python src/download_data.py

This gives 303 patients (Cleveland only). The Kaggle mirror linked in
data/README.md has all 920 patients from four hospitals and is the recommended
file; use this script only if you cannot download from Kaggle.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install the helper first:  pip install ucimlrepo") from exc

    dataset = fetch_ucirepo(id=45)  # UCI "Heart Disease"
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    out = ROOT / "data" / "heart_disease_cleveland.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print("Train with:  python src/train.py --data data/heart_disease_cleveland.csv")


if __name__ == "__main__":
    main()
