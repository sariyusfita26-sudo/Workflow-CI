"""
modelling.py
============
Script pelatihan model untuk MLflow Project.

Author  : Yusfitasari
Dataset : Heart Disease Dataset
Task    : Binary Classification
"""

import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

TARGET_COL = "target"


def load_data(train_path, test_path):
    print("[INFO] Memuat dataset...")
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    X_train  = train_df.drop(TARGET_COL, axis=1)
    y_train  = train_df[TARGET_COL]
    X_test   = test_df.drop(TARGET_COL, axis=1)
    y_test   = test_df[TARGET_COL]
    print(f"       Train : {X_train.shape}, Test : {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_confusion_matrix(y_test, y_pred, path="confusion_matrix.png"):
    cm   = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Sehat", "Sakit"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - Random Forest", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def save_roc_curve(y_test, y_pred_prob, path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score   = roc_auc_score(y_test, y_pred_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-AUC Curve - Random Forest", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def run(train_path, test_path, n_estimators, max_depth, random_state):
    print("=" * 60)
    print("  MODELLING - Heart Disease Classification")
    print("  Author: Yusfitasari | MLflow Project")
    print("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test = load_data(train_path, test_path)

    # Train
    print("[INFO] Melatih model...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy" : float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall"   : float(recall_score(y_test, y_pred)),
        "f1_score" : float(f1_score(y_test, y_pred)),
        "roc_auc"  : float(roc_auc_score(y_test, y_pred_prob)),
    }

    print("\n[INFO] Hasil Evaluasi:")
    for k, v in metrics.items():
        print(f"       {k:10s}: {v:.4f}")

    # Logging — gunakan active run dari MLflow Project
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth",    max_depth)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("train_size",   len(X_train))
    mlflow.log_param("test_size",    len(X_test))
    mlflow.log_param("n_features",   X_train.shape[1])
    mlflow.log_metrics(metrics)

    # Artefak
    cm_path  = save_confusion_matrix(y_test, y_pred)
    roc_path = save_roc_curve(y_test, y_pred_prob)
    mlflow.log_artifact(cm_path,  artifact_path="plots")
    mlflow.log_artifact(roc_path, artifact_path="plots")

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model"
    )

    mlflow.set_tag("author", "Yusfitasari")
    mlflow.set_tag("model",  "RandomForestClassifier")

    print("\n✅ Training selesai!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path",   type=str, default="heart_preprocessing/heart_train.csv")
    parser.add_argument("--test_path",    type=str, default="heart_preprocessing/heart_test.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth",    type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    run(
        train_path=args.train_path,
        test_path=args.test_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
