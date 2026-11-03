
# ==========================================================
# AIML ASSIGNMENT 2
# DIMENSIONALITY REDUCTION USING PCA
# ==========================================================

# 1. Import Necessary Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# 2. Load the Dataset

data = load_breast_cancer()

X = data.data
y = data.target

print("Original shape:", X.shape)

# Convert to DataFrame with feature names

df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

print("\nDataset:")
print(df.head())


# 3. Split Dataset into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Standardize the Features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nStandardized Features:")
print(X_train_scaled)


# 5. Calculate Covariance Structure

cov_matrix = np.cov(
    X_train_scaled,
    rowvar=False
)

print("\nCovariance Matrix:")
print(cov_matrix)


# 6. Apply PCA (2 Components)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_train_scaled)

print("\nBefore PCA:", X_train_scaled.shape)
print("After PCA:", X_pca.shape)


# 7. Explained Variance

print("\nExplained Variance:")
print(pca.explained_variance_ratio_)

total_variance = sum(pca.explained_variance_ratio_)

print("Variance retained:", total_variance)


# 8. Retaining 95% Variance

pca_95 = PCA(n_components=0.95)

X_train_pca = pca_95.fit_transform(X_train_scaled)
X_test_pca = pca_95.transform(X_test_scaled)

print("\nOriginal features:", X_train_scaled.shape[1])
print("Reduced features:", X_train_pca.shape[1])


# 9. Visualization Using First Two PCs

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y_train,
    cmap='viridis'
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Visualization")
plt.grid(True)
plt.show()


# 10. Classification Performance: Before PCA

start = time.time()

model = LogisticRegression(max_iter=5000)

model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)

accuracy_before = accuracy_score(y_test, pred)

end = time.time()

time_before = end - start

print("\nClassification Before PCA:")
print("Accuracy without PCA:", accuracy_before)
print("Execution time without PCA:", time_before, "seconds")


# 11. Classification Performance: After PCA

start = time.time()

model_pca = LogisticRegression(max_iter=5000)

model_pca.fit(X_train_pca, y_train)

pred_pca = model_pca.predict(X_test_pca)

accuracy_after = accuracy_score(y_test, pred_pca)

end = time.time()

time_after = end - start

print("\nClassification After PCA:")
print("Accuracy with PCA:", accuracy_after)
print("Execution time with PCA:", time_after, "seconds")


# 12. Final Comparison

print("\n========== FINAL COMPARISON ==========")

print("Original Features:", X_train_scaled.shape[1])
print("Reduced Features:", X_train_pca.shape[1])

print("\nAccuracy Before PCA:", accuracy_before)
print("Accuracy After PCA:", accuracy_after)

print("\nTime Before PCA:", time_before, "seconds")
print("Time After PCA:", time_after, "seconds")

print("\n========== PROGRAM COMPLETED ==========")
