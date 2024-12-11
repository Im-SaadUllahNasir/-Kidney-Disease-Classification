# Kidney Disease Classification

## Overview

This project focuses on the classification of **Kidney Disease** based on various medical attributes. Using machine learning techniques, the goal is to predict whether a patient has kidney disease or not based on features such as age, blood pressure, specific medical conditions, and other factors. The project demonstrates how we can use machine learning to assist healthcare professionals in diagnosing kidney-related issues early, improving patient outcomes.

This repository contains the code to train, evaluate, and test a classification model for kidney disease prediction.

## Dataset

The dataset used in this project is the **Chronic Kidney Disease (CKD) dataset**, which contains medical records for both healthy and diseased individuals. It includes various features such as:

- **Age**
- **Blood Pressure**
- **Specific Gravity**
- **Albumin**
- **Sugar**
- **Red Blood Cells**
- **Pus Cells**
- **Bacteria**
- **Blood Glucose Level**
- **Blood Urea**
- **Serum Creatinine**
- **Sodium**
- **Potassium**
- **Hemoglobin**
- **Hematocrit**
- **MCHC**
- **Tissue**

The dataset is available for download in CSV format and can be found in the `/data` directory (or linked in the dataset section).

### Example Dataset (Columns)

| Age | Blood Pressure | Specific Gravity | Albumin | Sugar | Red Blood Cells | Pus Cells | ... |
|-----|----------------|------------------|---------|-------|-----------------|-----------|-----|
| 45  | 80             | 1.020            | 1       | 0     | normal          | normal    | ... |
| 55  | 120            | 1.010            | 2       | 1     | normal          | abnormal  | ... |

## Features

- **Data Preprocessing**: The project includes steps for data cleaning, normalization, and feature engineering.
- **Classification Algorithms**: Machine learning algorithms such as **Logistic Regression**, **Random Forest**, **Support Vector Machine (SVM)**, and **K-Nearest Neighbors (KNN)** are used for classification.
- **Evaluation Metrics**: The model's performance is evaluated using metrics such as **Accuracy**, **Precision**, **Recall**, **F1-score**, and **AUC-ROC**.
- **Model Tuning**: Hyperparameter tuning techniques, such as **GridSearchCV**, are used to improve model performance.

## Installation

To run this project, you need Python 3.x and the required dependencies. You can set up the environment as follows:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Kidney-Disease-Classification.git
cd Kidney-Disease-Classification
```

### 2. Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all necessary libraries, such as `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and `xgboost`.

## Usage

### 1. **Data Preprocessing**

You can preprocess the dataset by running:

```bash
python preprocess.py
```

This script will clean the data (handling missing values, encoding categorical variables, etc.) and split the data into training and testing sets.

### 2. **Train the Model**

After preprocessing, you can train the model using the following command:

```bash
python train_model.py
```

This will train multiple classification models (e.g., Logistic Regression, Random Forest, etc.) and evaluate their performance.

### 3. **Evaluate the Model**

Once the model is trained, you can evaluate its performance on the test dataset using:

```bash
python evaluate_model.py
```

This will display various evaluation metrics such as accuracy, precision, recall, and F1-score.

### 4. **Prediction**

You can make predictions for new data by using the following command:

```bash
python predict.py --input data/new_patient_data.csv
```

Replace `new_patient_data.csv` with the path to your new patient data file.

## Example Output

After running the `train_model.py`, you should see output such as:

```bash
Model Accuracy: 95%
Precision: 0.94
Recall: 0.96
F1-Score: 0.95
```

## Model Evaluation

The model's performance is evaluated using several metrics, including:

- **Accuracy**: The overall accuracy of the model.
- **Precision**: The proportion of true positive predictions for kidney disease.
- **Recall**: The ability of the model to detect kidney disease.
- **F1-Score**: The balance between precision and recall.
- **ROC Curve & AUC**: For binary classification, the ROC curve and AUC score will also be plotted.

## Acknowledgments

- **Pandas** and **NumPy** for data manipulation.
- **Scikit-learn** for machine learning models and evaluation metrics.
- **Matplotlib** and **Seaborn** for data visualization.
- **XGBoost** for advanced boosting techniques (optional).

---
