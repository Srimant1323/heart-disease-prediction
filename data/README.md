# Data

Put the dataset here as **`data/heart_disease_uci.csv`**.

## Recommended: Kaggle mirror (all 920 patients)

1. Open https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data (free login required).
2. Click **Download**, unzip, and copy `heart_disease_uci.csv` into this folder.

It combines four hospitals (Cleveland, Hungary, Switzerland, VA Long Beach) and contains
real missing values, which makes the cleaning step meaningful.

## Alternative: UCI repository

Original source: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

If you cannot use Kaggle, `python src/download_data.py` fetches the Cleveland subset
(303 patients) through the `ucimlrepo` package. Then train with
`python src/train.py --data data/heart_disease_cleveland.csv`.

## Citation

Janosi, A., Steinbrunn, W., Pfisterer, M., Detrano, R. (1988). *Heart Disease* [Dataset].
UCI Machine Learning Repository. Creators: Hungarian Institute of Cardiology (Budapest),
University Hospital Zurich, University Hospital Basel, V.A. Medical Center Long Beach and
Cleveland Clinic Foundation.
