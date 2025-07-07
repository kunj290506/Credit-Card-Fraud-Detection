# Credit Card Fraud Detection

## Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset, `creditcard.csv`, contains 284,807 transactions with 31 features, including 30 numerical features (Time, V1–V28, Amount) and a binary Class label (0 for non-fraud, 1 for fraud). Due to the severe class imbalance (0.17% fraud cases), the project emphasizes preprocessing, exploratory data analysis (EDA), and building a robust K-Nearest Neighbors (KNN) classifier to identify fraud effectively.

## Objectives
- Clean and preprocess the dataset for consistency and quality.
- Perform exploratory data analysis to uncover patterns and insights.
- Apply feature engineering to enhance model performance.
- Develop and evaluate a machine learning model for fraud detection.
- Visualize findings and model performance for clear interpretation.

## Dataset
The dataset used is `creditcard.csv`, sourced from Kaggle. It contains:
- **Rows**: 284,807 transactions
- **Columns**: 31 (Time, V1–V28, Amount, and Class)
- **Class Distribution**: 284,315 non-fraud (0) and 492 fraud (1) cases
- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Methodology
1. **Data Loading and Inspection**:
   - Loaded the dataset using `pandas.read_csv()`.
   - Inspected shape (`df.shape`), columns (`df.columns`), data types, missing values (`df.info()`), and statistical summary (`df.describe()`).

2. **Data Preprocessing**:
   - Confirmed no missing values in the dataset.
   - Scaled numerical features (V1–V28, Amount) using `StandardScaler` for normalization.
   - Split data into 80% training and 20% testing sets using `train_test_split` with `random_state=42`.

3. **Exploratory Data Analysis (EDA)**:
   - Visualized class imbalance using `sns.countplot()`.
   - Analyzed feature distributions with histograms (`sns.histplot()`) and correlations with heatmaps (`sns.heatmap()`).

4. **Feature Engineering**:
   - Applied standard scaling to V1–V28 and Amount; no additional features created as V1–V28 are PCA-transformed.

5. **Model Building and Evaluation**:
   - Trained a K-Nearest Neighbors (KNN) classifier (`KNeighborsClassifier`, `random_state=42`) on the scaled training data.
   - Evaluated performance using `accuracy_score`, `confusion_matrix`, and `classification_report`.
   - Visualized the confusion matrix with `sns.heatmap()` to highlight True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

## Results
| Metric             | Value  |
|--------------------|--------|
| Accuracy           | 99.95% |
| True Negatives (TN)| 56,855 |
| False Positives (FP)| 7      |
| False Negatives (FN)| 18     |
| True Positives (TP)| 82     |
| Precision (Fraud)  | 0.92   |
| Recall (Fraud)     | 0.82   |
| F1-Score (Fraud)   | 0.87   |

### Key Insights
- High accuracy (99.95%) is driven by the dominant non-fraud class, making precision and recall for fraud cases critical.
- The KNN model detects 82% of fraud cases with minimal false positives (7), indicating strong performance.
- The F1-score (0.87) balances precision and recall, though 18 missed frauds suggest room for improvement.
- Future improvements could include hyperparameter tuning (e.g., `n_neighbors`), ensemble methods (e.g., Random Forest), or oversampling techniques like SMOTE.

## Conclusion
The K-Nearest Neighbors model provides a robust baseline for credit card fraud detection, effectively handling class imbalance with high precision and recall. Future enhancements could involve testing ensemble models, advanced sampling techniques, or hyperparameter optimization to further reduce false negatives while maintaining low false positives.

## Libraries Used
- `numpy`
- `pandas`
- `matplotlib.pyplot`
- `seaborn`
- `scikit-learn` (`StandardScaler`, `train_test_split`, `KNeighborsClassifier`, `accuracy_score`, `confusion_matrix`, `classification_report`)

## Requirements
To run the notebook, install the required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
