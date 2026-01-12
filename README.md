# Production Machine Failure Prediction

## Problem Statement

This project focuses on building and comparing multiple machine learning models to predict the failure of steel production machines. The objective is to predict a binary target variable based on a set of operational and sensor data sourced from the production plant.

A key challenge in this problem is the severe class imbalance present in the training data, which makes naive modeling approaches and standard accuracy-based evaluation unreliable.

The final goal is to generate predictions for an unseen test dataset that does not contain target labels.

---

## Dataset Overview

The project uses two datasets:

- `train.csv`
- `test.csv`

### Training Data

- Contains feature columns and a binary target variable
- Target distribution is highly imbalanced:
  - Approximately 98.4% class 0
  - Approximately 1.6% class 1
- This imbalance strongly influences model selection, validation strategy, and metric choice

### Test Data

- Contains the same feature columns as the training data
- Does not contain the target variable
- Used strictly for inference after model selection and training
- No evaluation metrics can be computed on this dataset

---

## Key Challenges

- Extreme class imbalance in the target variable
- Accuracy is not a meaningful evaluation metric
- Test data has no labels, requiring careful validation using training data only
- Need for fair model comparison without data leakage

---

## Preprocessing Summary

- Selected continuous numerical features were scaled using `StandardScaler`
- Ordinal categorical features were encoded but not scaled
- Feature and target separation was performed before any scaling
- Scaling parameters were learned only from training data and reused for test data
- Column names were sanitized to ensure compatibility with all models, especially XGBoost

---

## Validation Strategy

To ensure robust and unbiased evaluation under class imbalance:

- Stratified cross-validation was used throughout
- Each fold preserved the original class distribution
- This prevented folds with missing or extremely rare positive samples

---

## Models Trained

The following models were trained and evaluated:

### Logistic Regression

- Used as a strong baseline model
- Class imbalance handled using `class_weight="balanced"`
- Sensitive to feature scaling
- Provides interpretable coefficients

### Random Forest Classifier

- Tree-based ensemble model
- Naturally handles non-linear relationships
- Class imbalance handled using `class_weight="balanced"`
- Feature scaling not required

### XGBoost Classifier

- Gradient boosting based tree ensemble
- Designed for high-performance tabular modeling
- Class imbalance handled using `scale_pos_weight`
- Feature scaling not required but allowed
- Strong ranking performance under imbalance

---

## Hyperparameter Tuning

- Hyperparameters were optimized using `GridSearchCV`
- Stratified cross-validation was used during tuning
- Each model was tuned independently
- Only training data was used during hyperparameter selection

---

## Evaluation Metrics

Due to the extreme class imbalance, accuracy was not used as a primary evaluation metric.

The following metrics were used:

- **ROC-AUC**
  - Primary metric for model selection
  - Measures ranking quality across all decision thresholds
- **F1 Score**
  - Reported to evaluate the balance between precision and recall
  - Computed using cross-validated estimates

Each model was optimized using ROC-AUC and then evaluated using both ROC-AUC and F1 score for comparison.

![Training Performance of the models](training/training-model-performance.png)

_**NOTE**_: This isn't the final model performance since that requires the actual target values in `test.csv`, which this dataset didn't have.

---

## Model Performance Summary

All models demonstrated strong ranking ability with high ROC-AUC values, indicating that meaningful signal was learned despite the imbalance.

Logistic Regression achieved the highest F1 score under the default threshold, while Random Forest and XGBoost achieved superior ranking performance but lower F1 scores, highlighting the need for threshold tuning in ensemble models.

---

## Final Inference

- The best estimator for each model was refit on the full training dataset
- Trained models were applied to the test dataset
- Predictions and predicted probabilities were stored in a structured DataFrame
- No evaluation metrics were computed on the test set due to missing labels

---

## Key Takeaways

- Severe class imbalance requires careful metric selection
- Stratified cross-validation is essential for reliable evaluation
- ROC-AUC is effective for model selection under imbalance
- F1 score provides complementary insight into classification performance
- High ROC-AUC with lower F1 is expected without threshold tuning
- Test data must be treated strictly as unseen data for inference

---
