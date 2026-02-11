# Customer Churn Prediction

This project focuses on predicting customer churn using supervised machine learning models.
The goal is to identify customers who are likely to stop using a service, which can help businesses take preventive actions.

## Problem Overview
Customer churn is a key business problem, especially in subscription-based services.
Using historical customer data, this project builds classification models to predict whether a customer will churn.

## Dataset
The dataset contains customer-level information used for training and testing the models.
It includes features related to customer behavior and service usage.

Files:
- `training.xlsx` – training dataset
- `testing.xlsx` – test dataset

## Models Used
The following machine learning models were implemented and compared:
- Decision Tree Classifier
- Random Forest Classifier

Hyperparameter tuning was performed using GridSearchCV with cross-validation.

## Evaluation Metrics
Model performance was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

These metrics were chosen to better assess model performance on imbalanced data.

## Technologies
- Python
- pandas
- scikit-learn
- matplotlib

## Project Structure
- `customer-churn-prediction.py` – main script containing data preprocessing, model training, and evaluation
- `training.xlsx` – training data
- `testing.xlsx` – testing data

## Results
The Random Forest model achieved better overall performance compared to the Decision Tree,
especially in terms of recall and F1-score, making it more suitable for churn prediction.

## Conclusion
This project demonstrates an end-to-end machine learning workflow:
from data preprocessing, through model training and tuning, to evaluation.
It highlights the importance of choosing appropriate evaluation metrics for classification problems.

## Future Improvements
- Feature engineering and feature importance analysis
- Handling class imbalance with resampling techniques
- Adding model interpretability (e.g. SHAP values)
- Converting the project to a Jupyter Notebook for better presentation
