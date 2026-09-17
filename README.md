# Credit Card Fraud Detection using XGBoost

## Overview

This project focuses on detecting fraudulent credit card transactions using **XGBoost** on the real-world **IEEE-CIS Fraud Detection Dataset**.

Because credit card fraud datasets are highly imbalanced, traditional accuracy-based evaluation can be misleading. This project addresses the class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)** and evaluates the model using metrics such as **Precision, Recall, F1-Score, ROC-AUC, and PR-AUC**.

The project also performs **decision threshold tuning** to study the trade-off between detecting fraudulent transactions and generating false positives. Finally, **XGBoost feature importance scores** are used to identify which features contribute most to the model's predictions.

---

## Objectives

* Detect fraudulent credit card transactions.
* Handle severe class imbalance using SMOTE.
* Train an XGBoost classification model.
* Evaluate model performance using appropriate fraud-detection metrics.
* Optimize the classification decision threshold.
* Analyze feature importance using XGBoost.
* Understand the trade-off between false positives and false negatives.

---

## Dataset

The project uses the **IEEE-CIS Fraud Detection Dataset** available through Kaggle.

The dataset contains transaction information from a real-world fraud detection competition and includes the target variable:

```text
isFraud
```

where:

```text
0 = Legitimate transaction
1 = Fraudulent transaction
```

The dataset contains a large number of legitimate transactions compared with fraudulent transactions, making it a highly imbalanced classification problem.

Dataset source:

https://www.kaggle.com/c/ieee-fraud-detection

### Dataset Files

The competition provides files including:

```text
train_transaction.csv
train_identity.csv
test_transaction.csv
test_identity.csv
```

This project primarily uses:

```text
train_transaction.csv
```

The identity dataset can be incorporated in a future version by joining it with the transaction dataset using `TransactionID`.

---

## Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn
* Matplotlib

---

## Machine Learning Workflow

```text
IEEE-CIS Dataset
       ↓
Data Loading
       ↓
Feature Selection
       ↓
Missing Value Handling
       ↓
Train-Test Split
       ↓
SMOTE on Training Data
       ↓
XGBoost Classifier
       ↓
Fraud Probability Prediction
       ↓
Decision Threshold Tuning
       ↓
Model Evaluation
       ↓
Feature Importance Analysis
```

---

## Handling Class Imbalance

Fraudulent transactions represent only a small proportion of the dataset.

To address this problem, **SMOTE** is applied to the training data.

SMOTE creates synthetic examples of the minority class instead of simply duplicating existing fraud observations.

```python
smote = SMOTE(
    sampling_strategy=0.30,
    random_state=42,
    k_neighbors=5
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)
```

SMOTE is applied **after the train-test split** so that the test set remains representative of the original data distribution.

---

## XGBoost Model

The project uses XGBoost for binary classification.

Example configuration:

```python
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)
```

The model produces a probability of fraud for every transaction.

---

## Decision Threshold Tuning

Instead of relying only on the default classification threshold of `0.50`, multiple thresholds are tested.

For example:

```text
0.05
0.10
0.15
0.20
...
0.90
0.95
```

For each threshold, Precision, Recall, and F1-Score are calculated.

The threshold providing the highest F1-Score is selected for the final demonstration.

### Why Threshold Tuning?

Fraud detection involves a trade-off:

```text
Lower threshold
      ↓
More transactions classified as fraud
      ↓
Higher fraud detection / recall
      ↓
Potentially more false positives
```

Whereas:

```text
Higher threshold
      ↓
Fewer transactions classified as fraud
      ↓
Potentially fewer false positives
      ↓
Potentially more missed fraud
```

Therefore, the threshold should be selected according to the intended application and relative costs of false positives and false negatives.

---

## Evaluation Metrics

The following metrics are used:

### Accuracy

Measures the overall proportion of correctly classified transactions.

However, accuracy alone is not sufficient for highly imbalanced fraud datasets.

### Precision

```text
Precision = TP / (TP + FP)
```

Measures how many transactions predicted as fraud were actually fraudulent.

### Recall

```text
Recall = TP / (TP + FN)
```

Measures how many actual fraudulent transactions were successfully detected.

### F1-Score

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Provides a balance between Precision and Recall.

### ROC-AUC

Measures the model's ability to distinguish between fraudulent and legitimate transactions across different classification thresholds.

### PR-AUC

Measures performance through the Precision-Recall relationship and is particularly useful for evaluating heavily imbalanced classification problems.

---

## Confusion Matrix

The project generates a confusion matrix containing:

```text
                 Predicted
              Legitimate  Fraud
Actual
Legitimate       TN        FP
Fraud             FN        TP
```

Where:

* **TN** = Legitimate transactions correctly identified.
* **FP** = Legitimate transactions incorrectly classified as fraud.
* **FN** = Fraudulent transactions incorrectly classified as legitimate.
* **TP** = Fraudulent transactions correctly identified.

---

## Feature Importance

XGBoost feature importance is used to identify the features that contributed most to the model's predictions.

The project generates a visualization of the top 20 important features.

```python
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)
```

A high feature importance score indicates that the feature was useful to the trained model for making predictions. It does **not** imply that the feature causes fraudulent behavior.

---

## Results

The notebook calculates the following final metrics:

```text
Accuracy
Precision
Recall
F1-Score
ROC-AUC
PR-AUC
Decision Threshold
```

The exact results may vary depending on the selected features, preprocessing, train-test split, SMOTE configuration, XGBoost parameters, and computational environment.

Results generated by the notebook should be added here after running the complete experiment.

Example:

| Metric             |      Value |
| ------------------ | ---------: |
| Accuracy           | Add result |
| Precision          | Add result |
| Recall             | Add result |
| F1-Score           | Add result |
| ROC-AUC            | Add result |
| PR-AUC             | Add result |
| Decision Threshold | Add result |

---

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── Credit_Card_Fraud_Detection.ipynb
├── README.md
├── requirements.txt
│
└── data/
    └── README.md
```

The original dataset files are not included in this repository because of their large size and dataset distribution considerations.

---

## Installation

Install the required Python libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost imbalanced-learn
```

Or use the provided `requirements.txt` file.

---

## Running the Project

### Google Colab

1. Open the notebook in Google Colab.
2. Download the IEEE-CIS dataset from Kaggle.
3. Upload/configure your Kaggle API credentials.
4. Run the notebook cells sequentially.
5. The notebook will train the XGBoost model and generate the evaluation results and visualizations.

### Local Environment

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the IEEE-CIS dataset separately and update the dataset path in the notebook.

---

## Limitations

* The IEEE-CIS dataset contains many features and missing values, requiring substantial preprocessing and computational resources.
* A selected subset of numerical features is used in this implementation to keep SMOTE and model training manageable in Google Colab.
* SMOTE creates synthetic training observations and does not represent real fraudulent transactions.
* Feature importance indicates model usage of features, not causal relationships.
* The model is an academic machine learning project and should not be considered a production-ready banking fraud detection system.
* The optimal decision threshold depends on the costs associated with false positives and false negatives.

---

## Future Improvements

Possible improvements include:

* Incorporating `train_identity.csv`.
* Using additional transaction and identity features.
* Comparing XGBoost with LightGBM, Random Forest, and Logistic Regression.
* Using SHAP for more detailed model interpretation.
* Performing hyperparameter optimization.
* Evaluating cost-sensitive learning.
* Testing additional imbalance-handling techniques.
* Using time-based validation to better reflect real-world transaction prediction.
* Developing a real-time fraud detection API.

---

## Conclusion

This project demonstrates how **XGBoost**, **SMOTE**, and **decision threshold optimization** can be combined to address a highly imbalanced fraud detection problem.

Rather than relying solely on accuracy, the project focuses on Precision, Recall, F1-Score, ROC-AUC, and PR-AUC to provide a more meaningful evaluation of fraud detection performance.

The project also demonstrates how changing the classification threshold affects the balance between detecting fraudulent transactions and incorrectly flagging legitimate transactions.

---

## Author

**Vaibhav**

Machine Learning / CSE-AIML Student
