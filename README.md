# CodeAlpha Credit Scoring Model

## 📌 Project Title

**Credit Scoring Model using Machine Learning**

---

## 📖 Project Overview

This project builds a **Machine Learning model to predict a customer's credit score category (Good / Poor)** based on financial and behavioral attributes.

Credit scoring is widely used by **banks and financial institutions** to determine whether a customer is eligible for loans or credit cards.

The model analyzes customer financial history and predicts their **creditworthiness**, helping reduce financial risk.

---

## 🎯 Problem Statement

The objective of this project is to build a Machine Learning model capable of predicting whether a person is **creditworthy** based on their financial history and demographic details.

In the banking and lending industry, accurately predicting credit risk is crucial for:

* Reducing loan defaults
* Improving credit approval decisions
* Managing financial risk effectively

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Loading

The dataset is loaded using **Pandas** from the training and testing CSV files.

### 2️⃣ Data Cleaning

The dataset is cleaned by:

* Handling missing values
* Removing duplicates
* Encoding categorical variables

### 3️⃣ Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset.

Generated visualizations:

* Correlation Heatmap
* Feature Distribution Graphs

### 4️⃣ Feature Selection

Important features were selected to train the model.

### 5️⃣ Model Training

Multiple Machine Learning models were trained:

* Logistic Regression
* Decision Tree
* Random Forest

### 6️⃣ Model Evaluation

Models were evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

### 7️⃣ Predictions

The best performing model was used to generate predictions for the test dataset.

---

## 📊 Example Output

| ID     | Customer_ID | Predicted_Score |
| ------ | ----------- | --------------- |
| 0x160a | CUS_0xd40   | Good            |
| 0x160b | CUS_0xd40   | Good            |
| 0x160c | CUS_0xd40   | Good            |

Predictions are stored in:

```
final_predictions.csv
```

---

## 🗂 Project Structure

```
CodeAlpha_Credit_Scoring_Model
│
├── dataset
│   ├── train.csv
│   └── test.csv
│
├── credit_scoring_model.py
├── requirements.txt
├── README.md
│
├── correlation_heatmap.png
├── feature_distributions.png
└── final_predictions.csv
```

---

##  Installation

Install required libraries:

```
pip install -r requirements.txt
```

---

##  Run the Project

Run the model using:

```
python credit_scoring_model.py
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## Author

**Venu Gopal R**
B.Tech – Artificial Intelligence & Data Science

### How to Run:
```bash
pip install -r requirements.txt
python credit_scoring_model.py
```
