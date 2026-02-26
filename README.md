# Machine Failure Prediction (Predictive Maintenance)

## Problem Statement

Unplanned machine failures in industrial settings (e.g., manufacturing or oil & gas plants) lead to costly downtime and maintenance overhead.
The objective of this project is to build a **predictive maintenance model** that can **identify potential machine failures in advance** using sensor data, with a strong emphasis on **detecting rare failure events**.

Given the highly imbalanced nature of the data (~97% non-failure vs ~3% failure), the problem is framed as a **rare-event binary classification task**, where **missing failures (false negatives)** is more costly than triggering extra inspections (false positives).

---

## Dataset Description

### Source

The dataset used is the [**AI4I 2020 Predictive Maintenance Dataset**](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), sourced from **UCI Machine Learning Repository.</br>
This is a widely used public benchmark for predictive maintenance tasks.

---

### Columns Overview

| Column                  | Description                                   |
| ----------------------- | --------------------------------------------- |
| UDI                     | Unique row identifier                         |
| Product ID              | Machine / product identifier                  |
| Type                    | Product quality category (L, M, H)            |
| Air temperature [K]     | Ambient temperature                           |
| Process temperature [K] | Internal process temperature                  |
| Rotational speed [rpm]  | Machine rotational speed                      |
| Torque [Nm]             | Applied torque                                |
| Tool wear [min]         | Tool usage time                               |
| Machine failure         | Target variable (1 = failure, 0 = no failure) |
| TWF                     | Tool Wear Failure                             |
| HDF                     | Heat Dissipation Failure                      |
| PWF                     | Power Failure                                 |
| OSF                     | Overstrain Failure                            |
| RNF                     | Random Failure                                |

> Note: The individual failure-type columns were **excluded from modeling** to prevent target leakage.

---

## Preprocessing Summary

* Dropped identifier columns (`UDI`, `Product ID`)
* Encoded categorical feature `Type`
* Renamed columns for consistency and modeling convenience
* Used **stratified train–test split** to preserve failure distribution
* Applied **feature scaling** where required (for linear models)
* Ensured **no data leakage** by fitting preprocessing steps only on training data

---

## Models Trained & Training Approach

The following models were trained and compared using a **consistent evaluation pipeline**:

### Models

* Logistic Regression (baseline, class-weighted)
* Random Forest
* LightGBM

### Training Technique

* **Stratified K-Fold Cross-Validation (5 folds)**
* Hyperparameter tuning using `GridSearchCV`
* Model selection optimized using **PR-AUC**
* Final predictions generated using **probability threshold tuning** to align with operational objectives

---

## Evaluation Metrics

Given the severe class imbalance, traditional accuracy was avoided.

Primary metrics used:

* **PR-AUC (Precision–Recall AUC)** – model discrimination under imbalance
* **Recall (Failure class)** – ability to detect failures
* **Precision (Failure class)** – false alarm control
* **F1-score** – balance between precision and recall

![Scores across each model](Screenshots/scores.png)

---

## Performance Summary

| Model               | PR-AUC  | Recall         | Precision |
| ------------------- | ------- | -------------- | --------- |
| Logistic Regression | Low     | High           | Low       |
| Random Forest       | Medium  | Highest        | Moderate  |
| LightGBM            | Highest | Slightly Lower | Highest   |

* Logistic Regression produced a **pessimistic model** with many false positives.
* Random Forest improved balance by capturing **non-linear relationships**.
* LightGBM delivered the **best overall performance**, achieving strong precision while maintaining acceptable recall.

---

## Key Inferences & Takeaways

* **Model evaluation matters as much as model choice** in imbalanced classification problems.
* High recall with low precision reflects a **conservative failure-detection policy**, not a weak model.
* Tree-based models outperform linear models due to **non-linear sensor interactions**.
* **LightGBM emerged as the most suitable model** when balancing failure detection and operational efficiency.
* Threshold tuning is an **operational decision**, not a form of cheating, and is essential in predictive maintenance use cases.

---

## Final Outcome

This project demonstrates an **end-to-end predictive maintenance pipeline**, from data understanding to model evaluation and interpretation, highlighting how different modeling strategies impact the precision–recall tradeoff in rare-event prediction.

The final solution emphasizes **business-aligned metrics**, **honest evaluation**, and **model interpretability**, making it suitable as both a portfolio project and a real-world reference.
