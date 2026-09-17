# Kaggle Competition Solutions

Ongoing portfolio of Kaggle competition notebooks — EDA, feature engineering, modeling, and
post-competition write-ups. One folder per competition under `competitions/`.

## Goal

Consistent public presence: well-documented notebooks, discussion participation, and
competition submissions, building toward Kaggle Notebook Expert.

## Competition tracker

| Competition | Type | Started | Status | Best public score | Notebook |
|---|---|---|---|---|---|
| [playground-series-s6e9](competitions/playground-series-s6e9/) | Binary classification (tabular) | 2026-09-17 | In progress | — | [notebooks](competitions/playground-series-s6e9/notebooks/) |

## Structure per competition

```
competitions/<competition-slug>/
  README.md              problem, approach, results, key takeaways
  data/                   downloaded via `kaggle competitions download` (gitignored)
  notebooks/              01_eda, 02_feature_engineering, 03_modeling, ...
  submissions/            generated submission CSVs (gitignored)
```

## Setup

```bash
python3 -m venv ~/.venvs/kaggle-competition-solutions
source ~/.venvs/kaggle-competition-solutions/bin/activate
pip install -r requirements.txt
```

Each competition's data is downloaded on demand:

```bash
kaggle competitions download -c <competition-slug> -p competitions/<competition-slug>/data
```
