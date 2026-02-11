# Customer Churn Prediction (Decision Tree + Random Forest)

This project predicts customer churn (binary classification) using machine learning models:
a Decision Tree (baseline + tuned with GridSearchCV) and a Random Forest.
The workflow includes data loading, preprocessing (one-hot encoding), model training, evaluation, and saving outputs (plots + feature importances).

## What the script does
The main script:
1. Draws a simplified decision tree diagram (illustration) and saves it to `outputs/figures/simple_tree.png`
2. Loads training and testing datasets from Excel files (`training.xlsx`, `testing.xlsx`)
3. Preprocesses data:
   - drops `CustomerID`
   - separates target label `Churn`
   - one-hot encodes categorical features (`Gender`, `SubscriptionType`, `ContractLength`)
   - encodes train+test together to keep consistent dummy columns
4. Trains and evaluates:
   - **Decision Tree baseline** (`max_depth=4`)
   - **Decision Tree tuned** using **GridSearchCV** (scoring = F1, StratifiedKFold CV=3)
   - **Random Forest baseline** (`n_estimators=100`, `max_depth=8`)
5. Produces outputs:
   - decision tree plots saved in `outputs/figures/`
   - feature importances from Random Forest saved to `outputs/feature_importances.csv`

## Dataset & Features
The data is read from Excel files and parsed into columns:

**Target**
- `Churn` (0/1) — label used for prediction

**Dropped**
- `CustomerID` — removed from features

**Numeric features**
- `Age`
- `Tenure`
- `UsageFrequency`
- `SupportCalls`
- `PaymentDelay`
- `TotalSpend`
- `LastInteraction`

**Categorical features (one-hot encoded)**
- `Gender`
- `SubscriptionType`
- `ContractLength`

Files:
- `training.xlsx` — training data
- `testing.xlsx` — test data

## Models
### 1) Decision Tree (baseline)
- `DecisionTreeClassifier(max_depth=4, random_state=42)`

### 2) Decision Tree (tuned)
GridSearchCV parameters:
- `max_depth`: [3, 4, 5]
- `min_samples_split`: [10, 20, 40]
- `min_samples_leaf`: [10, 20, 30]

Cross-validation:
- `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`
Scoring:
- `f1`

### 3) Random Forest (baseline)
- `RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)`

## Evaluation
For each model, the script prints:
- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

## Outputs
Generated files:
- `outputs/figures/simple_tree.png` — simplified illustrative tree diagram
- `outputs/figures/base_tree.png` — plotted baseline decision tree (max_depth=4)
- `outputs/figures/best_tree.png` — plotted tuned decision tree (max_depth up to 5)
- `outputs/feature_importances.csv` — Random Forest feature importances (sorted)

## How to run
1. Put the files in the project folder:
   - `customer-churn-prediction.py`
   - `training.xlsx`
   - `testing.xlsx`

2. Install dependencies:
```bash
pip install pandas matplotlib scikit-learn openpyxl
# Customer Churn Prediction (Decision Tree + Random Forest)

This project predicts customer churn (binary classification) using machine learning models:
a Decision Tree (baseline + tuned with GridSearchCV) and a Random Forest.
The workflow includes data loading, preprocessing (one-hot encoding), model training, evaluation, and saving outputs (plots + feature importances).

## What the script does
The main script:
1. Draws a simplified decision tree diagram (illustration) and saves it to `outputs/figures/simple_tree.png`
2. Loads training and testing datasets from Excel files (`training.xlsx`, `testing.xlsx`)
3. Preprocesses data:
   - drops `CustomerID`
   - separates target label `Churn`
   - one-hot encodes categorical features (`Gender`, `SubscriptionType`, `ContractLength`)
   - encodes train+test together to keep consistent dummy columns
4. Trains and evaluates:
   - **Decision Tree baseline** (`max_depth=4`)
   - **Decision Tree tuned** using **GridSearchCV** (scoring = F1, StratifiedKFold CV=3)
   - **Random Forest baseline** (`n_estimators=100`, `max_depth=8`)
5. Produces outputs:
   - decision tree plots saved in `outputs/figures/`
   - feature importances from Random Forest saved to `outputs/feature_importances.csv`

## Dataset & Features
The data is read from Excel files and parsed into columns:

**Target**
- `Churn` (0/1) — label used for prediction

**Dropped**
- `CustomerID` — removed from features

**Numeric features**
- `Age`
- `Tenure`
- `UsageFrequency`
- `SupportCalls`
- `PaymentDelay`
- `TotalSpend`
- `LastInteraction`

**Categorical features (one-hot encoded)**
- `Gender`
- `SubscriptionType`
- `ContractLength`

Files:
- `training.xlsx` — training data
- `testing.xlsx` — test data

## Models
### 1) Decision Tree (baseline)
- `DecisionTreeClassifier(max_depth=4, random_state=42)`

### 2) Decision Tree (tuned)
GridSearchCV parameters:
- `max_depth`: [3, 4, 5]
- `min_samples_split`: [10, 20, 40]
- `min_samples_leaf`: [10, 20, 30]

Cross-validation:
- `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`
Scoring:
- `f1`

### 3) Random Forest (baseline)
- `RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)`

## Evaluation
For each model, the script prints:
- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

## Outputs
Generated files:
- `outputs/figures/simple_tree.png` — simplified illustrative tree diagram
- `outputs/figures/base_tree.png` — plotted baseline decision tree (max_depth=4)
- `outputs/figures/best_tree.png` — plotted tuned decision tree (max_depth up to 5)
- `outputs/feature_importances.csv` — Random Forest feature importances (sorted)

## How to run
1. Put the files in the project folder:
   - `customer-churn-prediction.py`
   - `training.xlsx`
   - `testing.xlsx`

2. Install dependencies:
```bash
pip install pandas matplotlib scikit-learn openpyxl
