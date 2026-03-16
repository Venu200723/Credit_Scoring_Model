# -*- coding: utf-8 -*-
import sys, io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import warnings
import joblib

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score, RocCurveDisplay
)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

warnings.filterwarnings("ignore")

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_TRAIN = os.path.join(BASE_DIR, "dataset", "train.csv")
DATASET_TEST  = os.path.join(BASE_DIR, "dataset", "test.csv")
MODEL_DIR     = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    print("\n[Step 1] Loading Dataset")
    try:
        df = pd.read_csv(DATASET_TRAIN, low_memory=False)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"  ERROR: '{DATASET_TRAIN}' not found.")
        sys.exit(1)


def eda(df, target_col):
    print("\n[Step 2] EDA")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"  Target distribution:\n{df[target_col].value_counts()}")

    numeric_df = df.select_dtypes(include=[np.number])

    if not numeric_df.empty:
        plt.figure(figsize=(12, 9))
        sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png", dpi=100)
        plt.close()
        print("  Saved → correlation_heatmap.png")

    cols_to_plot = [c for c in numeric_df.columns if c != target_col][:4]
    if cols_to_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, col in enumerate(cols_to_plot):
            sns.histplot(df[col].dropna(), kde=True, bins=30, ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")
        for j in range(len(cols_to_plot), 4):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.savefig("feature_distributions.png", dpi=100)
        plt.close()
        print("  Saved → feature_distributions.png")


def preprocess(df, target_col, fit=True,
               label_encoders=None, scaler=None,
               numeric_cols=None, categorical_cols=None,
               feature_cols=None):
    for col in df.columns:
        if df[col].dtype == "object":
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")
        else:
            med = df[col].median()
            df[col] = df[col].fillna(med if not pd.isna(med) else 0)

    df.drop_duplicates(inplace=True)

    if fit:
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        if df[target_col].dtype == "object":
            le_target = LabelEncoder()
            df[target_col] = le_target.fit_transform(df[target_col].astype(str))
            label_encoders[target_col] = le_target

        X = df.drop(columns=[target_col])
        y = df[target_col]

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        feature_cols = X.columns.tolist()

        return X, y, label_encoders, scaler, numeric_cols, categorical_cols, feature_cols

    else:
        for col in categorical_cols:
            if col in df.columns and col in label_encoders:
                le = label_encoders[col]
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else str(le.classes_[0])
                )
                df[col] = le.transform(df[col])

        if target_col in df.columns:
            df = df.drop(columns=[target_col])

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]

        df[numeric_cols] = scaler.transform(df[numeric_cols])
        return df


def train_and_evaluate(X_train, X_test, y_train, y_test, n_classes):
    print("\n[Step 3] Training & Cross-Validation")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    model = RandomForestClassifier(
        n_estimators=10, random_state=42, max_depth=10,
        class_weight="balanced", n_jobs=1
    )
    name = "Random Forest"

    print(f"\n  ── {name} ──")
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf,
                                 scoring="accuracy", n_jobs=1)
    print(f"  CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if n_classes == 2:
                roc = roc_auc_score(y_test, proba[:, 1])
            else:
                roc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        else:
            roc = float("nan")
    except Exception:
        roc = float("nan")

    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  ROC-AUC      : {roc:.4f}" if not np.isnan(roc) else "  ROC-AUC      : N/A")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"  Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

    results = {name: {"model": model, "accuracy": acc, "roc_auc": roc, "cv_mean": cv_scores.mean()}}
    return results, name, model, acc


def tune_best_model(best_model_name, best_model, X_train, y_train):
    print(f"\n[Step 4] Tuning: {best_model_name}")
    param_grid = {
        "n_estimators":     [10, 20],
        "max_depth":        [None, 10],
        "min_samples_split":[2, 10],
        "min_samples_leaf": [1, 4],
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        best_model, param_grid,
        n_iter=2, cv=skf,
        scoring="accuracy", n_jobs=1,
        random_state=42, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"  Best params  : {search.best_params_}")
    print(f"  Best CV score: {search.best_score_:.4f}")
    return search.best_estimator_


def save_feature_importance(model, feature_names, top_n=20):
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)

    if importance is None:
        return

    feat_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importance
    }).sort_values("Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=100)
    plt.close()
    print("  Saved → feature_importance.png")


def save_roc_curve(model, X_test, y_test, n_classes):
    if n_classes != 2 or not hasattr(model, "predict_proba"):
        return
    try:
        disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
        disp.figure_.savefig("roc_curve.png", dpi=100)
        plt.close()
        print("  Saved → roc_curve.png")
    except Exception as e:
        print(f"  ROC curve skipped: {e}")


def predict_test_file(model, label_encoders, scaler,
                      numeric_cols, categorical_cols,
                      feature_cols, target_col):
    print("\n[Step 6] Generating Final Predictions on dataset/test.csv")
    try:
        test_df = pd.read_csv(DATASET_TEST, low_memory=False)
        test_df_ids = test_df[["ID", "Customer_ID"]].copy() if all(
            c in test_df.columns for c in ["ID", "Customer_ID"]) else test_df.iloc[:, :2].copy()

        X_test_real = preprocess(
            test_df, target_col, fit=False,
            label_encoders=label_encoders, scaler=scaler,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            feature_cols=feature_cols
        )

        preds = model.predict(X_test_real)
        if target_col in label_encoders:
            preds = label_encoders[target_col].inverse_transform(preds)

        test_df_ids["Predicted_Score"] = preds
        test_df_ids.to_csv("final_predictions.csv", index=False)
        print("  Saved → final_predictions.csv")
        print(test_df_ids.head(10))
    except FileNotFoundError:
        print(f"  '{DATASET_TEST}' not found – skipping.")
    except Exception as e:
        print(f"  Prediction error: {e}")


def main():
    print("=" * 60)
    print("  CodeAlpha Credit Scoring Model  –  Improved Pipeline  ")
    print("=" * 60)

    df = load_data()
    target_col = df.columns[-1]
    print(f"\n  Target column: '{target_col}'")

    eda(df, target_col)

    print("\n[Step 3] Preprocessing")
    X, y, label_encoders, scaler, numeric_cols, categorical_cols, feature_cols = preprocess(
        df.copy(), target_col, fit=True
    )
    n_classes = y.nunique()
    print(f"  Features : {X.shape[1]}  |  Samples : {X.shape[0]}  |  Classes : {n_classes}")

    if SMOTE_AVAILABLE and n_classes >= 2:
        try:
            sm = SMOTE(random_state=42)
            X, y = sm.fit_resample(X, y)
            print(f"  SMOTE applied → new shape: {X.shape}")
        except Exception as e:
            print(f"  SMOTE skipped: {e}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    results, best_model_name, best_model, best_accuracy = train_and_evaluate(
        X_train, X_test, y_train, y_test, n_classes
    )

    print("\n" + "=" * 60)
    print(f"  Best Model : {best_model_name}")
    print(f"  Accuracy   : {best_accuracy:.4f}")
    print("=" * 60)

    tuned_model = tune_best_model(best_model_name, best_model, X_train, y_train)
    tuned_pred  = tuned_model.predict(X_test)
    tuned_acc   = accuracy_score(y_test, tuned_pred)
    print(f"\n  Tuned Model Accuracy: {tuned_acc:.4f}")

    final_model = tuned_model if tuned_acc >= best_accuracy else best_model
    print(f"  Final Model Accuracy: {max(tuned_acc, best_accuracy):.4f}")

    print("\n[Step 5] Saving Plots")
    save_feature_importance(final_model, feature_cols)
    save_roc_curve(final_model, X_test, y_test, n_classes)

    print("\n[Step 5] Saving Model Artifacts")
    joblib.dump(final_model,    os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler,         os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    joblib.dump({
        "numeric_cols":     numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_cols":     feature_cols,
        "target_col":       target_col,
        "n_classes":        n_classes,
        "best_model_name":  best_model_name,
        "accuracy":         max(tuned_acc, best_accuracy),
    }, os.path.join(MODEL_DIR, "metadata.pkl"))
    print(f"  Saved model artifacts to '{MODEL_DIR}/' folder.")

    predict_test_file(
        final_model, label_encoders, scaler,
        numeric_cols, categorical_cols, feature_cols, target_col
    )

    print("\n" + "=" * 60)
    print("  Pipeline Complete! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()