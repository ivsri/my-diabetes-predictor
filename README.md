# Diabetes Predictor — ML Report Helper

A small project that trains three classification models (Logistic Regression, Random Forest and XGBoost) on the "Diabetes Health Indicators Dataset" and prints report-ready outputs for inclusion in a written report.

This repository contains a single runnable script, `main.py`, which will:

- check for `diabetes_health_indicators_dataset.csv` in the project root and, if missing, attempt to download it from Kaggle using the `kaggle` API;
- train three models with class-imbalance handling (class_weight / scale_pos_weight);
- print three report-ready outputs to stdout:
  - `OUTPUT FOR REPORT: TABLE 1` — class balance counts/percentages
  - `OUTPUT FOR REPORT: TABLE 2` — model performance table (markdown)
  - `OUTPUT FOR REPORT: FIGURE 1` — feature importance printed and saved as `diabetes_figure1.png`

---

## Quick setup

Recommended: create and activate a Python virtual environment in the project root.

On macOS / zsh:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install required packages:

```bash
python -m pip install --upgrade pip
python -m pip install pandas matplotlib seaborn scikit-learn xgboost kaggle tabulate
```

If you prefer a requirements file, you can generate one from the active environment.

## Kaggle API (automatic download)

If you want the script to download the dataset automatically, create a Kaggle API token and place it at `~/.kaggle/kaggle.json` with secure permissions:

1. Go to https://www.kaggle.com/ -> My Account -> Create New API Token. This will download `kaggle.json`.
2. Move it to `~/.kaggle/kaggle.json` and set permissions:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

If you don't want to use the API, download `diabetes_health_indicators_dataset.csv` manually from the dataset page and place it in the project root.

## Run the script

With the virtualenv active:

```bash
python main.py
```

The script will print the three report outputs to stdout. The feature importance figure will be saved as `diabetes_figure1.png` in the project root.

## Files and outputs

- `main.py` — the main script.
- `diabetes_health_indicators_dataset.csv` — dataset (not committed, please add manually or let the script download it).
- `diabetes_figure1.png` — saved figure produced by the script.
- `.gitignore` — already configured to ignore `.venv/` and `*.csv` and python cache.

If you want the image and report tables saved to files instead of printing to stdout, I can add an option/flag for that.

## Notes

- The script attempts to auto-detect the target column name (supports `Diabetes_binary`, `diagnosed_diabetes`, `diabetes_stage`, etc.) and uses numeric columns only as features.
- The Random Forest feature importances are used for the Figure 1 output.
- Large files (dataset, virtualenv, image) should not be committed; confirm `.gitignore` excludes them before pushing.

## License

This project is released under the MIT License. Feel free to adapt for your report.

---

If you'd like, I can also:
- add a `requirements.txt` or `pyproject.toml` for reproducible installs;
- add a small CLI wrapper to save the three report outputs to files (e.g., `table1.txt`, `table2.md`, `diabetes_figure1.png`);
- prepare a cleaned commit (remove tracked large files) and push to a new GitHub repo for you.

Tell me which of the above you'd like next.
