# Credit Card Fraud Detection
Author: Chauhan Kunj Kirankumar

ID: D24AIML082

Date: July 06, 2025

## Project Overview
This notebook analyzes credit card transaction data to identify fraudulent activities using the `creditcard.csv` dataset. The key objectives are:

* Clean and preprocess the data to ensure quality and consistency.
* Apply feature engineering to enhance model performance.
* Conduct exploratory data analysis (EDA) to uncover patterns and insights.
* Develop and evaluate machine learning models to detect fraud.
* Visualize findings and model performance for clear interpretation.

## Dataset Download
To acquire the `creditcard.csv` dataset, I manually downloaded it from Kaggle. The dataset was retrieved by accessing the Kaggle website, locating the credit card fraud detection dataset, and downloading the `creditcard.csv` file directly to the local environment. This file is then placed in the current runtime directory, making it readily available for analysis in this notebook.

## Required Library Imports
To support the analysis and modeling tasks in this credit card fraud detection project, several essential Python libraries are utilized. Below is an explanation of the core libraries imported for this notebook:

* **NumPy**: Used for efficient numerical computations, enabling operations on arrays and matrices, which are critical for data manipulation and mathematical calculations in the dataset.
* **Pandas**: Employed for data handling and preprocessing, providing powerful tools to load, clean, and transform the `creditcard.csv` dataset into a structured format suitable for analysis.
* **Matplotlib**: Utilized for creating static visualizations, such as plots and charts, to illustrate patterns and model performance during exploratory data analysis and evaluation.
* **Seaborn**: A visualization library built on Matplotlib, used to generate enhanced, aesthetically pleasing graphs, such as heatmaps and distribution plots, to better understand the data and relationships.

These libraries form the foundation for data processing, analysis, and visualization throughout the project, ensuring efficient and effective execution of all tasks.

# Loading the Dataset
The `creditcard.csv` dataset is loaded into the notebook using the Pandas library's `read_csv` function, which efficiently reads the CSV file into a DataFrame for further analysis. To ensure all columns of the dataset are visible, the Pandas option `set_option('display.max_columns', None)` is applied, allowing the full structure of the data to be displayed without truncation. Finally, the DataFrame is displayed to provide an initial overview of the dataset, including its columns and sample rows, facilitating a quick understanding of its structure and contents.

## Checking Dataset Shape
To understand the scale of the `creditcard.csv` dataset, I examined its shape, which reveals the total number of rows and columns. Using the `df.shape` attribute, we found the dataset contains 284,807 rows and 31 columns. This large volume, with 284,807 transactions and 31 features (including Time, V1 to V28, Amount, and Class), helps us gauge the dataset's size and plan for memory and computational requirements in our analysis.

## Listing All Column Names
To gain a clear understanding of the features in the `creditcard.csv` dataset, I retrieve the names of all columns. By using the `df.columns` attribute, we obtain a list of all column headers, which helps identify the available variables, such as Time, V1 to V28, Amount, and Class. This step is essential for recognizing the dataset's structure and the features that will be used for analysis and modeling.

## Dataset Info Summary
To understand the structure and properties of the `creditcard.csv` dataset, I utilize the `df.info()` function, which provides essential metadata. This includes the count of non-null entries, the data types for each column, and the dataset's memory usage. This step is vital for detecting any missing values and determining the nature of the features available for analysis.

The dataset consists of 31 columns, all of which are numerical (including float and integer types) with no missing values, confirming a complete dataset suitable for subsequent processing and modeling.

## Checking Missing Values
To ensure the integrity of the `creditcard.csv` dataset, I check for the presence of any missing values. Using the `df.isnull().sum().sum()` method, which calculates the sum of null values across all columns, we confirm that there are no missing entries (0 missing values). This indicates a clean dataset, ready for direct use without requiring imputation or handling of missing data.

## Statistical Summary
To gain a comprehensive overview of the `creditcard.csv` dataset's numerical features, I generated a statistical summary using `df.describe()`. This function provides key descriptive statistics for each column, including the count, mean, standard deviation, minimum, maximum, and quartile values. This summary helps to quickly understand the distribution, central tendency, and spread of the data, which is crucial for identifying potential outliers, understanding feature scales, and assessing data quality before modeling.

## Class Distribution
To examine the balance of the `Class` variable in the `creditcard.csv` dataset, I analyzed its value counts using `df['Class'].value_counts()`. This revealed a highly imbalanced distribution:

* **Class 0 (Non-Fraudulent):** 284,315 transactions
* **Class 1 (Fraudulent):** 492 transactions

Fraudulent transactions constitute only a tiny fraction (approximately 0.17%) of the total dataset. This extreme imbalance is a critical challenge for fraud detection models, as it can lead to models that perform well on the majority class but poorly on the minority (fraud) class. Addressing this imbalance is crucial for building effective fraud detection systems.

## Data Visualization: Class Distribution Bar Plot
To visually represent the imbalance in the `Class` distribution, I created a bar plot using Seaborn's `countplot`. This plot clearly illustrates the overwhelming number of non-fraudulent transactions (Class 0) compared to the very small number of fraudulent transactions (Class 1). The extreme disparity between the two bars highlights the severe class imbalance challenge inherent in this credit card fraud detection dataset, reinforcing the need for specialized handling techniques during model development.

## Data Visualization: Transaction Amount Distribution
To understand the distribution of transaction amounts, I created a histogram using Matplotlib. This visualization shows the frequency of different transaction values. Most transactions are concentrated at lower amounts, with a long tail extending to higher values, indicating a skewed distribution. This insight helps in understanding the typical transaction behavior and identifies the range of values models need to handle.

## Data Visualization: Time Distribution
To visualize the distribution of transaction times, I created a histogram using Matplotlib. This plot shows the frequency of transactions over the 'Time' feature, which represents the seconds elapsed between each transaction and the first transaction in the dataset. The distribution appears relatively uniform across the observed time period, with some peaks and troughs that might correspond to varying transaction volumes throughout the day or specific time intervals. This helps in understanding the temporal patterns of the data, which could be useful for time-series analysis or feature engineering.

## Scaling the 'Amount' and 'Time' Features
To ensure that the 'Amount' and 'Time' features do not disproportionately influence the machine learning model due to their larger scales, I applied `StandardScaler` from `sklearn.preprocessing`.

* **'Amount' Scaling:** The 'Amount' column was scaled to have a mean of 0 and a standard deviation of 1.
* **'Time' Scaling:** Similarly, the 'Time' column was scaled using the same method.

The original 'Amount' and 'Time' columns were then dropped from the DataFrame, and the new scaled columns ('Scaled_Amount' and 'Scaled_Time') were added. This standardization is crucial for algorithms sensitive to feature magnitudes, like K-Nearest Neighbors, ensuring fair contribution from all features.

## Dropping Original 'Time' and 'Amount' Columns
After scaling the 'Time' and 'Amount' features and adding their scaled versions (`Scaled_Time` and `Scaled_Amount`) to the DataFrame, the original 'Time' and 'Amount' columns were dropped. This step ensures that the model only uses the normalized versions of these features, preventing any undue influence from their raw scales and maintaining data consistency for training.

## Splitting the Data into Features (X) and Target (y)
To prepare the dataset for model training, I split it into features (`X`) and the target variable (`y`).

* **Features (X):** All columns except the 'Class' column (which represents fraudulent or non-fraudulent transactions) were selected as features.
* **Target (y):** The 'Class' column was designated as the target variable, which the machine learning model will predict.

This separation is a standard practice in supervised learning, allowing the model to learn patterns from the input features to predict the corresponding output class.

## Splitting Data into Training and Testing Sets
To evaluate the model's performance on unseen data and prevent overfitting, I split the dataset into training and testing sets using `train_test_split` from `sklearn.model_selection`.

* **Training Set (80%):** Used to train the machine learning model.
* **Testing Set (20%):** Used to evaluate the trained model's performance.

The `stratify=y` parameter was used to ensure that both the training and testing sets have the same proportion of fraudulent and non-fraudulent transactions as the original dataset, which is crucial given the severe class imbalance. A `random_state` was set for reproducibility.

## K-Nearest Neighbors (KNN) Model Implementation
### Model Training
The K-Nearest Neighbors (KNN) classifier was implemented using `KNeighborsClassifier` from `sklearn.neighbors`.
* An instance of `KNeighborsClassifier` was created with `n_neighbors=5`, meaning the model considers the 5 nearest data points to classify a new point.
* The model was then trained using the `fit` method on the training data (`X_train`, `y_train`).

### Model Prediction
After training, the KNN model was used to make predictions on the test set (`X_test`) using the `predict` method. The predicted labels were stored in `y_pred`.

### Model Evaluation
To evaluate the performance of the KNN model, the following metrics and visualizations were used:

1.  **Accuracy Score:** Calculated using `accuracy_score` from `sklearn.metrics` to determine the overall correctness of predictions.
2.  **Confusion Matrix:** Generated using `confusion_matrix` from `sklearn.metrics` to visualize the counts of true positive, true negative, false positive, and false negative predictions.
3.  **Classification Report:** Generated using `classification_report` from `sklearn.metrics` to provide a detailed summary of precision, recall, F1-score, and support for each class (fraudulent and non-fraudulent).

These evaluations provide a comprehensive understanding of the model's effectiveness, especially in handling the imbalanced nature of credit card fraud detection.

## Results
* **Accuracy:** 0.9995
* **Confusion Matrix:**
    * True Negatives (non-fraud correctly predicted): 56860
    * False Positives (non-fraud incorrectly predicted as fraud): 5
    * False Negatives (fraud incorrectly predicted as non-fraud): 18
    * True Positives (fraud correctly predicted): 82

* **Classification Report:**
    * **Class 0 (Non-Fraud):**
        * Precision: 1.00
        * Recall: 1.00
        * F1-score: 1.00
    * **Class 1 (Fraud):**
        * Precision: 0.92
        * Recall: 0.82
        * F1-score: 0.87
* **Key Observations:**
    * The model achieves very high overall accuracy, largely due to the dominant non-fraud class, making precision and recall on fraud cases more meaningful.
    * The KNN model detected 82% of fraud cases with very few false positives, making it a strong baseline for fraud detection.
    * Precision (0.92) and F1-score (0.87) for fraud cases demonstrate good performance, although 18 missed frauds show room for improvement.
    * Techniques such as hyperparameter tuning (e.g., adjusting `n_neighbors`), ensemble methods, or data-level solutions like SMOTE could further boost detection rates.

### Conclusion
The K-Nearest Neighbors model offers a robust starting point for credit card fraud detection, balancing sensitivity and precision effectively despite class imbalance. Future work will focus on improving fraud detection by experimenting with ensemble models like Random Forest, advanced sampling methods, and parameter optimization to reduce missed fraud cases while maintaining a low false positive rate.

### Libraries Used
* numpy
* pandas
* matplotlib.pyplot
* seaborn
* sklearn (StandardScaler, train_test_split, KNeighborsClassifier, accuracy_score, confusion_matrix, classification_report)
