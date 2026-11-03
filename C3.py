
# ==========================================================
# GROUP C - ASSIGNMENT 3
# ADVANCED MACHINE LEARNING AND MODEL OPTIMIZATION
# SVM, RANDOM FOREST AND ADABOOST
# ==========================================================

# 1. IMPORT NECESSARY LIBRARIES

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================================
# 2. LOAD THE DATASET
# ==========================================================

# Load Breast Cancer Wisconsin Dataset

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset Loaded Successfully!")
print("Dataset Shape:", X.shape)

# ==========================================================
# 3. SPLIT THE DATASET
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ==========================================================
# 4. INITIALIZE CLASSIFICATION MODELS
# ==========================================================

models = {

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC())
    ]),

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "AdaBoost": AdaBoostClassifier(
        random_state=42
    )
}

# ==========================================================
# 5. TRAIN MODELS AND PERFORM CROSS-VALIDATION
# ==========================================================

baseline_results = []
cv_results = {}

print("\n========== BEFORE OPTIMIZATION ==========")

for name, model in models.items():

    # 5-Fold Cross Validation
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    cv_results[name] = scores.mean()

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    baseline_results.append({
        "Model": name,
        "CV Accuracy": scores.mean(),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print("\nModel:", name)
    print("Cross Validation Accuracy:", round(scores.mean(), 4))
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))


# ==========================================================
# 6. HYPERPARAMETER TUNING USING GRID SEARCH
# ==========================================================

param_grids = {

    "SVM": {
        "svm__C": [0.1, 1, 10],
        "svm__kernel": ["linear", "rbf"],
        "svm__gamma": ["scale", "auto"]
    },

    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    },

    "AdaBoost": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 1.0]
    }
}

optimized_models = {}
optimized_results = []

print("\n========== AFTER OPTIMIZATION ==========")

for name, model in models.items():

    print("\nOptimizing:", name)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    # Fit Grid Search
    grid_search.fit(X_train, y_train)

    # Best Model
    best_model = grid_search.best_estimator_

    optimized_models[name] = best_model

    # Predictions
    y_pred = best_model.predict(X_test)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    optimized_results.append({
        "Model": name,
        "CV Accuracy": grid_search.best_score_,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Accuracy:", round(grid_search.best_score_, 4))
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))


# ==========================================================
# 7. COMPARE MODEL PERFORMANCE
# ==========================================================

before_df = pd.DataFrame(baseline_results)
after_df = pd.DataFrame(optimized_results)

print("\n========== PERFORMANCE BEFORE OPTIMIZATION ==========")
print(before_df.to_string(index=False))

print("\n========== PERFORMANCE AFTER OPTIMIZATION ==========")
print(after_df.to_string(index=False))


# ==========================================================
# 8. COMBINED COMPARISON TABLE
# ==========================================================

comparison = pd.concat(
    [
        before_df.set_index("Model").add_prefix("Before "),
        after_df.set_index("Model").add_prefix("After ")
    ],
    axis=1
)

print("\n========== FINAL COMPARISON TABLE ==========")
print(comparison.round(4).to_string())


# ==========================================================
# 9. IDENTIFY THE BEST PERFORMING MODEL
# ==========================================================

best_index = after_df["Accuracy"].idxmax()

best_model_name = after_df.loc[best_index, "Model"]
best_accuracy = after_df.loc[best_index, "Accuracy"]

print("\n========== BEST PERFORMING MODEL ==========")

print("Best Model:", best_model_name)
print("Best Test Accuracy:", round(best_accuracy, 4))


# ==========================================================
# 10. CONCLUSION
# ==========================================================

print("\n========== CONCLUSION ==========")

print(
    "SVM, Random Forest and AdaBoost algorithms were "
    "implemented successfully."
)

print(
    "Five-fold cross-validation and Grid Search were "
    "used for model optimization."
)

print(
    "The models were evaluated using Accuracy, Precision, "
    "Recall and F1-Score."
)

print(
    "The best performing optimized model is:",
    best_model_name
)

print("\nProgram Executed Successfully!")
