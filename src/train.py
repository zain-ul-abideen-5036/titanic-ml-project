"""
src/train.py  —  Train, evaluate, and save the model.
Run:  python src/train.py
"""
import os, sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix)
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_for_training

DATA_PATH  = "data/titanic.csv"
MODEL_PATH = "artifacts/model.pkl"

def load_data(path):
    if not os.path.exists(path):
        print(f"\nERROR: {path} not found.")
        print("Download train.csv from https://www.kaggle.com/competitions/titanic/data")
        print("Rename it titanic.csv and place it in the data/ folder.\n")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df

def section(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")

def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    print(f"\n--- {name} ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred,
          target_names=["Did not survive","Survived"]))

def train():
    section("1. Load Data")
    df = load_data(DATA_PATH)

    section("2. Preprocess")
    X, y = preprocess_for_training(df)
    print(f"Features: {list(X.columns)}")
    print(f"Survival rate: {y.mean():.2%}")

    section("3. Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    section("4. Baseline — Logistic Regression")
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)
    evaluate(lr, X_test, y_test, "Logistic Regression")

    section("5. RandomForest + GridSearchCV")
    param_grid = {
        "n_estimators"   : [100, 200],
        "max_depth"      : [4, 6, 8, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf" : [1, 2],
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                        param_grid, cv=5, scoring="roc_auc",
                        n_jobs=-1, verbose=1)
    print("Running GridSearchCV …")
    grid.fit(X_train, y_train)
    print(f"Best params : {grid.best_params_}")
    print(f"Best CV AUC : {grid.best_score_:.4f}")
    best = grid.best_estimator_

    section("6. Final Evaluation")
    evaluate(best, X_test, y_test, "Best RandomForest")

    section("7. Feature Importances")
    for i in np.argsort(best.feature_importances_)[::-1]:
        bar = "█" * int(best.feature_importances_[i] * 50)
        print(f"  {X.columns[i]:<15} {best.feature_importances_[i]:.4f}  {bar}")

    section("8. Save Model")
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")
    print("\nTraining complete!\n")

if __name__ == "__main__":
    train()
