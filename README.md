# NSL-KDD Intrusion Detection

This project replicates the paper _Machine learning-based intrusion detection
framework using NSL-KDD dataset_, published in **Discover Artificial
Intelligence** by Springer Nature in 2026. The paper is fully open access:

- Paper: <https://link.springer.com/article/10.1007/s44163-026-01930-9>
- Dataset: NSL-KDD
- Kaggle download: <https://www.kaggle.com/datasets/hassan06/nslkdd>
- Original UNB source: <https://www.unb.ca/cic/datasets/nsl.html>

NSL-KDD contains 148,517 records: 125,973 training records and 22,544 test
records. Each record has 41 network-traffic features and a class label. The
project treats the task as binary intrusion detection: `normal` traffic is the
negative class and every attack type is the positive class.

The replication evaluates Logistic Regression, K-Nearest Neighbors, Naive
Bayes, Decision Tree, Random Forest, and a proposed stacking classifier. The
reported metrics include accuracy, precision, recall, F1 score, and ROC-AUC.

## Setup

Run everything inside a local Python virtual environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Quick smoke test:

```powershell
python replicate_nslkdd.py --merged-csv data\NSL-KDD_merged.csv --sample 20000
```

Here, `--merged-csv` tells `replicate_nslkdd.py` to use the existing merged
file and run the paper protocol. It does not merge files and it is not needed
again when `data\NSL-KDD_merged.csv` already exists.

Full paper-protocol run, all models including the proposed stacking model:

```powershell
python replicate_nslkdd.py --merged-csv data\NSL-KDD_merged.csv
```

Save the fitted stacking model and preprocessing as a PKL:

```powershell
python replicate_nslkdd.py --merged-csv data\NSL-KDD_merged.csv --save-pkl
```

The default output is `results\nslkdd_stacking_proposed.pkl`. Use `--pkl-out path\to\model.pkl` to choose a different path.

Build the official-protocol bundle for the Streamlit app:

```powershell
python export_model.py
streamlit run app_streamlit.py
```

`export_model.py` expects the official files `data\KDDTrain+.csv` and
`data\KDDTest+.csv` by default. It writes the fitted inference bundle to
`model\nslkdd_model.pkl` and its evaluation metrics to `model\metrics.json`.
The Streamlit app loads that bundle and provides a form for classifying one
network connection at a time. It also displays the official-protocol metrics
stored in the bundle and can randomize test inputs.

If the files are stored elsewhere, pass their paths explicitly:

```powershell
python export_model.py --train path\to\KDDTrain+.csv --test path\to\KDDTest+.csv
streamlit run app_streamlit.py
```

Keep the terminal running while using the app, then open the local URL shown by
Streamlit (normally `http://localhost:8501`). Dependencies for both scripts
are in `requirements.txt`.

### What `export_model.py` exports

The script does **not** train several models and select the one with the best
test score. It always trains the project's official stacking model using:

- Random Forest (`n_estimators=100`)
- Decision Tree
- K-Nearest Neighbors (`n_neighbors=5`)
- Bernoulli Naive Bayes
- Logistic Regression as the final estimator

The base models are combined by `StackingClassifier` with five-fold
cross-validation and probability-based stacking. The script fits the scaler on
the official training data and fits categorical encoders using categories from
both official files, evaluates the resulting fixed stacking model on the
official test data, and exports the model together with encoders, scaler,
feature lists, input defaults/ranges, and evaluation metrics. Therefore,
`model\nslkdd_model.pkl` is the **proposed stacking model**, not an automatically
chosen “best model.”

The two original files can also be used directly with the same paper protocol:

```powershell
python replicate_nslkdd.py --train data\KDDTrain+.csv --test data\KDDTest+.csv
```

Deactivate the environment when finished:

```powershell
deactivate
```
