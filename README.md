# 💳 CodeAlpha Credit Scoring Model

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Neural%20Network-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-ff6b6b?style=for-the-badge&logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A Deep Learning–powered Credit Scoring System with an interactive Gradio Web UI**

</div>

---

## 📖 Project Overview

This project builds a **Deep Learning Neural Network model** to predict a customer's **credit score category (Good / Standard / Poor)** based on financial and behavioural attributes.

Credit scoring is used by **banks and financial institutions** to determine whether a customer is eligible for loans or credit cards. This model analyses customer financial history and predicts their **creditworthiness**, helping reduce financial risk.

---

## 🎯 Problem Statement

Build a machine learning pipeline that predicts whether a person is **creditworthy** based on their financial history and demographic details.

**Why it matters:**
- ✅ Reduce loan defaults
- ✅ Improve credit approval decisions
- ✅ Manage financial risk effectively
- ✅ Provide explainable, confidence-backed predictions

---

## 🧠 Model Architecture — Deep Neural Network (MLP)

The model uses **scikit-learn's `MLPClassifier`** with the following design:

| Component | Details |
|---|---|
| **Architecture** | 4 hidden layers: `256 → 128 → 64 → 32` neurons |
| **Activation** | ReLU (non-linearity) |
| **Optimizer** | Adam with adaptive learning rate |
| **Regularisation** | L2 (alpha = 0.001 → 0.005 after tuning) |
| **Early Stopping** | Yes — patience = 20 epochs |
| **Batch Size** | 64 |
| **Validation Split** | 10% held out during training |
| **Cross-Validation** | 2-fold Stratified K-Fold |

---

## 🔄 Machine Learning Pipeline

```
Raw CSV Data
    │
    ▼
┌─────────────────┐
│  1. Load Data   │  ── train.csv & test.csv
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. EDA         │  ── Correlation Heatmap, Feature Distributions
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  3. Preprocessing    │  ── Fill NaN, Encode, Scale (StandardScaler)
│     + SMOTE (opt.)   │  ── Handle class imbalance if available
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  4. Train MLP        │  ── Deep Neural Network (256→128→64→32)
│     + Cross-Val      │  ── 2-Fold Stratified CV
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  5. Hyperparameter   │  ── Refined alpha=0.005, lr=0.0005
│     Tuning           │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  6. Evaluation       │  ── Accuracy, ROC-AUC, Confusion Matrix
│     + Plots          │  ── Loss Curve, Feature Importance, ROC Curve
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  7. Save Artifacts   │  ── best_model.pkl, scaler.pkl, metadata.pkl
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  8. Predict          │  ── final_predictions.csv from test.csv
└──────────────────────┘
```

---

## 📊 Outputs & Visualisations

| Output File | Description |
|---|---|
| `correlation_heatmap.png` | Heatmap showing feature correlations |
| `feature_distributions.png` | Distribution plots for top numeric features |
| `feature_importance.png` | Permutation importance bar chart (top 20) |
| `training_loss_curve.png` | Training loss + validation accuracy over epochs |
| `roc_curve.png` | ROC curve (binary classification) |
| `final_predictions.csv` | Predicted credit scores for test dataset |

### 📋 Sample Predictions

| ID | Customer_ID | Predicted_Score |
|---|---|---|
| 0x160a | CUS_0xd40 | Good |
| 0x160b | CUS_0xd40 | Standard |
| 0x160c | CUS_0xd41 | Poor |

---

## 🖥️ Interactive Web UI (Gradio)

The project includes a **Gradio-powered web dashboard** (`app.py`) with:

- 🎛️ **Input sliders and dropdowns** for all financial features
- 🎯 **Real-time credit score prediction** (Good / Standard / Poor)
- 📊 **Confidence chart** — probability breakdown per class
- 📌 **Feature importance chart** — top 10 most influential features
- 🌙 **Dark theme** with gradient UI design

### UI Input Fields

| Section | Fields |
|---|---|
| 👤 Personal Info | Name, Age, Monthly Income, Occupation |
| 💳 Credit Behaviour | Credit Mix, Min Amount Paid, Delayed Payments, Days Past Due |
| 📊 Financial Details | Outstanding Debt, Credit Utilization %, Interest Rate, Credit History Age, No. of Credit Inquiries |

---

## 🗂️ Project Structure

```
CodeAlpha_Credit_Scoring_Model/
│
├── dataset/
│   ├── train.csv                   # Training data
│   └── test.csv                    # Test data
│
├── model/                          # Auto-generated after training
│   ├── best_model.pkl              # Trained MLP model
│   ├── scaler.pkl                  # StandardScaler
│   ├── label_encoders.pkl          # LabelEncoders for categorical features
│   └── metadata.pkl                # Feature names, accuracy, importances
│
├── credit_scoring_model.py         # Full deep learning training pipeline
├── app.py                          # Gradio web UI
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── correlation_heatmap.png         # EDA output
├── feature_distributions.png       # EDA output
├── feature_importance.png          # Permutation importance chart
├── training_loss_curve.png         # Loss curve during training
├── roc_curve.png                   # ROC curve (binary)
└── final_predictions.csv           # Test set predictions
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Venu200723/CodeAlpha_Credit_Scoring_Model.git
cd CodeAlpha_Credit_Scoring_Model
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Train the Model

```bash
python credit_scoring_model.py
```

This will:
- Load and preprocess the dataset
- Train the Deep Neural Network
- Save model artifacts to `model/`
- Generate all visualisation plots
- Output `final_predictions.csv`

### Step 2 — Launch the Web UI

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:7860**

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `Python 3.8+` | Core language |
| `Pandas` | Data manipulation |
| `NumPy` | Numerical operations |
| `Scikit-learn` | MLPClassifier, preprocessing, evaluation |
| `Imbalanced-learn` | SMOTE for class balancing (optional) |
| `Matplotlib` | Plotting and visualisations |
| `Seaborn` | Statistical visualisations |
| `Gradio` | Interactive web UI |
| `Joblib` | Model serialisation |

---

## 📈 Model Evaluation Metrics

- ✅ **Accuracy Score**
- ✅ **ROC-AUC Score** (macro-average for multi-class)
- ✅ **Confusion Matrix**
- ✅ **Classification Report** (Precision, Recall, F1)
- ✅ **2-Fold Stratified Cross-Validation**
- ✅ **Permutation Feature Importance**

---

## 👨‍💻 Author

**Venu Gopal R**  
B.Tech – Artificial Intelligence & Data Science

[![GitHub](https://img.shields.io/badge/GitHub-Venu200723-181717?style=flat&logo=github)](https://github.com/Venu200723)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

<div align="center">
  <sub>Built with ❤️ as part of the CodeAlpha Internship Program</sub>
</div>
