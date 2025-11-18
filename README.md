# User Analytics & Churn Prediction Platform

This repository contains an end-to-end machine learning project for **customer churn prediction** on a subscription-like product.

It is built to look like a small real-world ML platform:

- Reproducible **data pipeline** (raw → cleaned → train/val/test splits)
- Proper **feature engineering** for numeric and categorical data
- Several **models** (logistic regression, random forests, XGBoost, MLP with PyTorch)
- **Hyperparameter tuning**, ROC curves, and fair model comparison
- **Data analysis / insights** to understand *why* customers churn (not just a black-box model)
- An interactive **Streamlit dashboard** for KPIs, segment analysis and customer-level scoring

The project is generic enough to be adapted to many domains:
telecom, subscription apps, games, fintech, streaming, etc.

---

## 1. Dataset

The project uses the public **Telco Customer Churn** dataset (IBM sample).

Each row represents one customer and includes:

- Demographics (e.g. gender, senior citizen, dependents)
- Contract and billing information (contract type, payment method, paperless billing, tenure)
- Services (phone, internet, streaming, security, support, etc.)
- Charges (`MonthlyCharges`, `TotalCharges`)
- Target column `Churn` with values:
  - `"Yes"` → customer churned
  - `"No"` → customer stayed

During preprocessing, a numeric target column is created:

- `ChurnFlag = 1` → churned  
- `ChurnFlag = 0` → not churned  

Files layout:

- Raw data:  
  `data/raw/telco_customer_churn.csv`
- Cleaned data:  
  `data/interim/telco_clean.csv`
- Splits:  
  `data/processed/telco_train.csv`  
  `data/processed/telco_val.csv`  
  `data/processed/telco_test.csv`

Global churn rate in this dataset is around **26.5%**.

---

## 2. Project Structure

```text
user-analytics-churn-platform/
├── data/
│   ├── raw/                  # raw CSV
│   ├── interim/              # cleaned data
│   └── processed/            # train / val / test splits
├── models/                   # saved model files (*.joblib, PyTorch weights)
├── reports/
│   ├── figures/              # ROC curves, feature importance plots
│   ├── feature_importances/  # feature importance tables
│   └── *.csv                 # churn analysis tables (by contract, tenure, etc.)
├── dashboards/
│   └── streamlit_app/
│       └── app.py            # interactive churn analytics dashboard (Streamlit)
├── src/
│   └── churn_platform/
│       ├── data/             # loading, preprocessing, splitting
│       ├── features/         # feature building (X, y, numeric/categorical)
│       ├── models/           # baselines, tuning, evaluation, persistence, DL models
│       ├── analysis/         # churn profiling and summary tables
│       ├── visualization/    # plotting utilities (EDA, ROC, feature importance)
│       └── cli/              # command-line entrypoints (run_*.py)
├── tests/                    # (optional unit tests)
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 3. Setup

### 3.1. Create and activate a virtual environment

Example (Windows PowerShell):

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Or on Linux/Mac:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3.2. Install dependencies

```bash
pip install -r requirements.txt
```

Make sure the raw dataset is placed at:

```text
data/raw/telco_customer_churn.csv
```

### 3.3. Add `src/` to `PYTHONPATH` for CLI usage

From the project root:

```bash
# PowerShell
$env:PYTHONPATH = "$PWD\src"

# bash
export PYTHONPATH="$PWD/src"
```

---

## 4. End-to-end Pipeline

### 4.1. Exploratory Data Analysis (EDA)

```bash
python -m churn_platform.cli.run_eda
```

This will:

- Load the raw CSV
- Print basic info and column types
- Plot:
  - Distribution of the target (`Churn`)
  - Distributions of key numeric features
  - Some relationships between features and churn

All EDA code lives in `src/churn_platform/visualization/`.

---

### 4.2. Preprocessing and train/val/test split

```bash
python -m churn_platform.cli.run_preprocess
```

This step:

- Cleans the data (types, missing values, numeric target `ChurnFlag`)
- Drops duplicates based on `customerID`
- Converts `TotalCharges` to numeric and fills missing values with median
- Creates **stratified** train/validation/test splits:

  - Train: 4225 rows  
  - Validation: 1409 rows  
  - Test: 1409 rows  

Cleaned and split CSVs are saved under `data/interim/` and `data/processed/`.

Key functions:

- `clean_telco_data` in `data/preprocess.py`
- `train_val_test_split` in `data/split.py`

---

### 4.3. Feature engineering

All basic feature logic is in:

- `src/churn_platform/features/build_features.py`

Main idea:

- Input: a cleaned `DataFrame` (e.g. train split)
- Output:
  - `X`: feature matrix (drop `ChurnFlag`, `customerID`, `Churn`)
  - `y`: target vector (`ChurnFlag`)
  - `numeric_cols`: numeric feature names
  - `categorical_cols`: categorical feature names

Preprocessing for models uses a `ColumnTransformer`:

- `StandardScaler` for numeric features
- `OneHotEncoder(handle_unknown="ignore")` for categorical features

---

### 4.4. Train baseline models

#### Logistic Regression baseline

```bash
python -m churn_platform.cli.run_training
```

This trains a pipeline:

- Preprocessor (scaler + one-hot)
- Logistic regression (`class_weight="balanced"`, `max_iter=1000`)

Model is evaluated on the validation set and saved to:

```text
models/logreg_baseline.joblib
```

#### Random Forest baseline

```bash
python -m churn_platform.cli.run_training_rf
```

This trains a random forest baseline with similar preprocessing and saves:

```text
models/rf_baseline.joblib
```

#### XGBoost baseline

```bash
python -m churn_platform.cli.run_training_xgb
```

This trains an `XGBClassifier` with default-ish hyperparameters and saves:

```text
models/xgb_baseline.joblib
```

---

### 4.5. Hyperparameter tuning (Random Forest)

```bash
python -m churn_platform.cli.run_tuning_rf
```

This runs a `RandomizedSearchCV` over a random forest pipeline:

- Varies `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- Uses 3-fold CV and optimizes **ROC AUC**
- Prints best hyperparameters and validation metrics

The best model is saved as:

```text
models/rf_tuned.joblib
```

---

### 4.6. Deep learning model (MLP with PyTorch)

A simple feed-forward neural network is implemented in:

- `models/dl_models.py` (`SimpleMLP`)

Training script:

```bash
python -m churn_platform.cli.run_training_dl
```

This script:

- Builds a **separate** preprocessor (StandardScaler + OneHotEncoder)
- Encodes features into a dense NumPy array
- Converts data to PyTorch tensors
- Trains an MLP with:
  - 2 hidden layers, ReLU activations, dropout
  - `BCEWithLogitsLoss` with `pos_weight` to handle class imbalance
  - Adam optimizer
- Evaluates on the validation set, prints metrics, and searches for the best decision threshold for F1

Artifacts saved:

- Preprocessor: `models/mlp_preprocessor.joblib`
- MLP weights: `models/mlp_model.pt`

This model is later used for ROC comparison.

---

### 4.7. Final evaluation on the test set

You can evaluate any saved model on the test set by editing:

- `src/churn_platform/cli/run_evaluation.py`

Choose the model file, for example:

```python
from churn_platform.models.persistence import load_model

# Example: tuned random forest
model = load_model(filename="rf_tuned.joblib")
```

Then run:

```bash
python -m churn_platform.cli.run_evaluation
```

This will:

- Load `data/processed/telco_test.csv`
- Build features
- Load the chosen model
- Print classification report, confusion matrix, and metrics.

---

## 5. Data Analysis and Churn Insights

Beyond model training, the project computes **churn profiling** tables.

Script:

```bash
python -m churn_platform.cli.run_analysis
```

This:

- Loads train + val + test
- Concatenates everything into a single dataset
- Computes churn rates by:
  - contract type
  - binned tenure
  - binned total charges
- Saves results into multiple CSV files under `reports/`.

### 5.1. Global churn rate

From `global_churn_rate.csv`:

- Overall churn rate ≈ **26.5%**

This means roughly 1 out of 4 customers churn in this dataset.

### 5.2. Churn by contract type

From `reports/churn_by_Contract.csv`:

| Contract        | Count | Churn rate |
|----------------|-------|-----------:|
| Month-to-month | 3875  | ~42.7%     |
| One year       | 1473  | ~11.3%     |
| Two year       | 1695  | ~2.8%      |

**Interpretation:**

- Customers on **month-to-month** contracts churn a lot:
  - ~42.7% churn rate.
- **One-year** contracts are much more stable (~11.3%).
- **Two-year** contracts are extremely stable (~2.8%).

Month-to-month churn is about **4× higher** than one-year and about **15× higher** than two-year contracts.

> Insight: contract structure has a huge impact on churn; long-term contracts strongly reduce churn.

### 5.3. Churn by tenure (time with the company)

From `reports/churn_by_tenure_bin.csv`:

| Tenure bin       | Count | Churn rate |
|------------------|-------|-----------:|
| 0–6 months       | 1481  | ~52.9%     |
| 6–20 months      | 1397  | ~33.4%     |
| 20–40 months     | 1408  | ~22.4%     |
| 40–60 months     | 1350  | ~15.6%     |
| 60–72 months     | 1407  | ~6.6%      |

**Interpretation:**

- New customers (0–6 months) have **very high churn** (~53%).
- Churn decreases steadily as tenure increases.
- Very long-tenure customers (60–72 months) churn very little (~6.6%).

> Insight: the first months of the customer lifecycle are the most critical for retention.

### 5.4. Churn by total charges (proxy for lifetime value)

From `reports/churn_by_TotalCharges_bin.csv`:

| TotalCharges bin      | Count | Churn rate |
|-----------------------|-------|-----------:|
| ~18.8–267.4           | 1409  | ~46.0%     |
| 267.4–947.4           | 1408  | ~28.9%     |
| 947.4–2043.7          | 1409  | ~20.7%     |
| 2043.7–4471.4         | 1408  | ~23.0%     |
| 4471.4–8684.8         | 1409  | ~14.1%     |

Overall pattern: **lower total charges → higher churn**, higher total charges (longer, deeper relationships) → lower churn.

> Insight: low total charges (newer or low-engagement customers) are much more likely to churn than high-value long-term customers.

---

## 6. Model Comparison

Several models were trained and compared, all using the same feature pipeline
(StandardScaler for numeric data, OneHotEncoder for categorical data).

### 6.1. Models

1. **Logistic Regression baseline**
   - Simple linear model
   - `class_weight="balanced"`

2. **Random Forest baseline**
   - Tree-based ensemble
   - `n_estimators=300`, `class_weight="balanced"`

3. **Random Forest tuned**
   - RandomizedSearchCV over RF hyperparameters
   - Example best params:
     - `n_estimators = 500`
     - `max_depth = 20`
     - `min_samples_split = 10`
     - `min_samples_leaf = 8`
     - `max_features = "sqrt"`

4. **XGBoost baseline**
   - `XGBClassifier` with:
     - `n_estimators=300`
     - `max_depth=5`
     - `learning_rate=0.1`
     - `subsample=0.8`
     - `colsample_bytree=0.8`
     - `tree_method="hist"`

5. **MLP (PyTorch)**
   - Simple feed-forward neural network
   - 2 hidden layers, ReLU, dropout
   - Trained with `BCEWithLogitsLoss` and `pos_weight` for class imbalance
   - Uses a separate preprocessor (scaler + one-hot) and dense input features

### 6.2. Test set metrics (selected models)

Approximate metrics on the **test set** (churn = positive class):

| Model                | Accuracy | Precision | Recall | F1    | ROC AUC |
|----------------------|---------:|----------:|-------:|------:|--------:|
| Logistic Regression  | 0.7395   | 0.5060    | 0.7834 | 0.6149| 0.8426  |
| RF baseline          | 0.7821   | 0.6143    | 0.4813 | 0.5397| 0.8195  |
| RF tuned             | 0.7580   | 0.5311    | 0.7540 | 0.6232| 0.8421  |
| XGBoost baseline     | 0.7921   | 0.6302    | 0.5241 | 0.5723| 0.8225  |

On the validation set, the MLP (PyTorch) achieves performance similar to the logistic regression baseline (F1 ~0.61, ROC AUC ~0.83), showing that a simple neural network can match classical models on this tabular dataset.

### 6.3. ROC curves

The script:

```bash
python -m churn_platform.cli.run_plot_roc_curves
```

- Computes ROC curves on the test set for:
  - Logistic Regression
  - Random Forest baseline
  - Random Forest tuned
  - XGBoost baseline
  - MLP (PyTorch)
- Saves the combined ROC figure to:

```text
reports/figures/roc_all_models.png
```

In this plot:

- Logistic Regression and the tuned Random Forest reach **AUC ≈ 0.84**
- MLP (PyTorch) is close behind with AUC ≈ 0.83
- XGBoost and RF baseline are slightly lower in AUC but still clearly better than random

> Conclusion: several models perform well in terms of ranking churners vs non-churners; the tuned random forest and logistic regression provide the best AUC, with different trade-offs between accuracy and recall.

### 6.4. Final model choice

Given a churn use case, **recall on churners is important** (missing a churner is costly), but overall accuracy also matters.

- Logistic Regression:
  - High recall on churners
  - Strong AUC
- Random Forest tuned:
  - Better balance between accuracy and recall
  - AUC almost identical to logistic regression
- XGBoost:
  - Best accuracy, but lower recall on churners

In this project, the **tuned Random Forest (`rf_tuned`)** is selected as the *final model*, with:

- **Competitive AUC (~0.84)**
- **Good recall on churners (~0.75)**
- **Reasonable accuracy (~0.76)**

Logistic regression, XGBoost, and the MLP are kept as strong comparative baselines.

---

## 7. Feature Importance and Interpretation

To understand *why* the model makes its predictions, we analyse feature importances
from the tuned random forest and XGBoost.

Script:

```bash
python -m churn_platform.cli.run_feature_importance
```

This:

- Loads `rf_tuned.joblib` and `xgb_baseline.joblib`
- Extracts feature importances (after preprocessing)
- Saves tables to:
  - `reports/feature_importances/feature_importances_rf_tuned.csv`
  - `reports/feature_importances/feature_importances_xgb_baseline.csv`
- Generates bar plots under:
  - `reports/figures/feature_importances_rf_tuned.png`
  - `reports/figures/feature_importances_xgb_baseline.png`

### 7.1. Top features (tuned Random Forest)

From `feature_importances_rf_tuned.csv`, the most important features include:

- `cat__Contract_Month-to-month`
- `num__tenure`
- `num__TotalCharges`
- `cat__Contract_Two year`
- `num__MonthlyCharges`
- `cat__OnlineSecurity_No`
- `cat__TechSupport_No`
- `cat__InternetService_Fiber optic`
- `cat__PaymentMethod_Electronic check`

**Interpretation:**

- **Contract type** is critical:
  - Being on a month-to-month contract is strongly associated with higher churn.
  - Two-year contracts push in the opposite direction (lower churn).
- **Tenure** and **TotalCharges**:
  - Short tenure and low total charges correspond to higher churn.
  - Long tenure and high total charges correspond to more loyal customers.
- **MonthlyCharges**:
  - Higher monthly charges are associated with a higher risk of churn (especially combined with low tenure).
- **Service and support features**:
  - `OnlineSecurity_No` and `TechSupport_No` appear as important:
    customers without security/support services tend to churn more.
- **Internet and payment type**:
  - Fiber optic internet and electronic check payments are correlated with higher churn in this dataset.

These feature importances match the earlier churn analysis:
short-tenure, month-to-month, high-charge customers with weaker service bundles
are the most at risk.

---

## 8. Streamlit Dashboard

The project includes a simple **Streamlit dashboard** to explore churn KPIs,
segments and individual customer risk.

### 8.1. Launching the dashboard

From the project root, with your virtual environment activated:

```bash
# Make sure src/ is in PYTHONPATH
$env:PYTHONPATH = "$PWD\src"   # PowerShell
# or
export PYTHONPATH="$PWD/src"    # bash

streamlit run dashboards/streamlit_app/app.py
```

This will open a browser window at `http://localhost:8501/`.

### 8.2. Tabs and functionality

The dashboard has three main tabs:

1. **Overview**
   - Global KPIs:
     - global churn rate
     - number of customers
   - Churn rate by **contract type** (table + bar chart)
   - Distribution of **tenure** (histogram)

2. **Segments**
   - Interactive segmentation of churn by:
     - `Contract`
     - `tenure` (binned into quantiles)
     - `TotalCharges` (binned into quantiles)
   - For each segment:
     - number of customers
     - churn rate and churn rate (%)  
   - Visualisation as tables and bar charts.

3. **Customer scoring**
   - Select an existing `customerID` from the **test set**
   - Display the raw customer data (JSON view)
   - Compute churn probability using the **tuned Random Forest pipeline**
   - Choose a decision threshold (slider)
   - Display:
     - predicted churn probability
     - predicted label (`Churn` / `No churn`)
     - true label (for that customer), so you can see if the model is correct

This dashboard demonstrates how the model can be turned into a small internal tool
for analysts / product managers to explore churn and manually inspect risk
for specific customers.

---

## 9. Possible Extensions

Some natural next steps for future work:

- Add **calibration** and custom decision thresholds per segment
- Add **partial dependence plots / SHAP** for more detailed interpretability
- Allow manual editing of customer features in the dashboard to see how risk changes
- Add basic **tests** in `tests/` (e.g. checking that preprocessing and splitting behave as expected)

---

## 10. Summary

This project demonstrates how to build a **production-style churn prediction pipeline**:

- Clean data processing and splitting
- Serious model comparison: logistic regression, random forests, XGBoost, and an MLP
- Hyperparameter tuning and fair evaluation on a held-out test set
- ROC curves and feature importances to understand model behavior
- Concrete **insights** on which customers are at risk:
  - month-to-month contracts,
  - short tenure,
  - low total charges,
  - high monthly charges,
  - missing security/support services.
- An interactive **Streamlit dashboard** that exposes KPIs, segment-level churn, and customer-level scoring.

It can be used as a template for user analytics and churn modelling in many product contexts
(games, apps, telecom, finance, streaming, etc.).
