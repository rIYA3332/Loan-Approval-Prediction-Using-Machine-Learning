# Loan Prediction: Logistic Regression vs Decision Tree vs Random Forest

A binary classification project that predicts whether a loan application 
will be approved or rejected, comparing three machine learning models on 
the Loan Prediction dataset.

## Dataset
- Source: Loan Prediction Dataset via HuggingFace
- Task: Predict whether a loan will be Approved (Y) or Rejected (N)
- 614 records with 13 features including income, credit history, 
  loan amount, and property area

## Models Used
- Logistic Regression with TF-IDF features
- Decision Tree with interpretable decision paths
- Random Forest ensemble method

## Results

| Model               | CV Accuracy |
|---------------------|-------------|
| Random Forest       | ~82%        |
| Decision Tree       | ~79%        |
| Logistic Regression | ~80%        |

Random Forest achieved the highest cross-validation score, showing 
that ensemble methods outperform single models on this dataset.

## What the notebook covers
- Exploratory Data Analysis with univariate, bivariate and 
  multivariate analysis
- Missing value handling using mode and median imputation
- Outlier treatment using log transformation
- Feature engineering: EMI, Total Income, Balance Income
- Class imbalance handling using SMOTE
- Hyperparameter tuning using GridSearchCV with Stratified K-Fold
- Evaluation using Accuracy, Precision, Recall, F1, ROC Curve, 
  Confusion Matrix and Precision-Recall Curve

## Libraries
Python, Scikit-learn, Imbalanced-learn, Pandas, NumPy, 
Matplotlib, Seaborn, SciPy
