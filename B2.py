
# ==========================================================
# AIML PRACTICAL 2 - REGRESSION TECHNIQUES
# Linear Regression and Polynomial Regression
# Dataset: California Housing Dataset (housing.csv)
# ==========================================================

# 1. IMPORTING REQUIRED LIBRARIES

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 2. LOADING THE DATASET

df = pd.read_csv("housing.csv")

print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())


# 3. DATA PREPROCESSING

# Checking dimensions
print("\nDataset Shape:")
print(df.shape)

# Printing information
print("\nDataset Information:")
df.info()

# Descriptive statistics
print("\nStatistical Description:")
print(df.describe())

# Checking missing values
print("\nMissing Values:")
print(df.isnull().sum())


# 4. HANDLING MISSING VALUES

df['total_bedrooms'] = df['total_bedrooms'].fillna(
    df['total_bedrooms'].median()
)


# 5. DETERMINING CATEGORICAL FEATURES

print("\nOcean Proximity:")
print(df['ocean_proximity'].value_counts())


# 6. ENCODING CATEGORICAL FEATURES

ohe = OneHotEncoder(
    sparse_output=False,
    drop='first'
)

encoded = ohe.fit_transform(df[['ocean_proximity']])

encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(['ocean_proximity']),
    index=df.index
)

df = pd.concat(
    [df.drop('ocean_proximity', axis=1), encoded_df],
    axis=1
)

print("\nDataset After Encoding:")
print(df.head())


# 7. EXPLORATORY DATA ANALYSIS

# Finding pairwise correlation
print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# Generating correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# 8. SCATTER PLOT

sns.scatterplot(
    x='median_income',
    y='median_house_value',
    data=df
)

plt.title("Median Income vs Median House Value")
plt.show()


# 9. BUILDING LINEAR REGRESSION MODEL

# Extracting independent and dependent variables

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']


# Splitting dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Creating Linear Regression model

lin_reg = LinearRegression()


# Fitting Linear Regression model

lin_reg.fit(X_train, y_train)


# Predicting test set results

y_pred = lin_reg.predict(X_test)


# 10. EVALUATING LINEAR REGRESSION

print("\n========== LINEAR REGRESSION ==========")

print("MSE:", mean_squared_error(y_test, y_pred))

print("MAE:", mean_absolute_error(y_test, y_pred))

print("R2 Score:", r2_score(y_test, y_pred))


# 11. VISUALIZATION OF LINEAR REGRESSION

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=X_test['median_income'],
    y=y_test,
    alpha=0.3,
    label="Test Data"
)

sns.lineplot(
    x=X_test['median_income'],
    y=y_pred,
    color='red',
    label="Predicted Values"
)

plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Linear Regression Visualization")
plt.show()


# 12. POLYNOMIAL REGRESSION

# Creating Polynomial Features

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)


# Creating Polynomial Regression model

poly_reg = LinearRegression()


# Fitting Polynomial Regression model

poly_reg.fit(X_train_poly, y_train)


# Predicting test set results

y_pred_poly = poly_reg.predict(X_test_poly)


# 13. EVALUATING POLYNOMIAL REGRESSION

print("\n========== POLYNOMIAL REGRESSION ==========")

print("MSE:", mean_squared_error(y_test, y_pred_poly))

print("MAE:", mean_absolute_error(y_test, y_pred_poly))

print("R2 Score:", r2_score(y_test, y_pred_poly))


# 14. COMPARISON OF BOTH MODELS

print("\n========== MODEL COMPARISON ==========")

print("\nLinear Regression:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

print("\nPolynomial Regression:")
print("MSE:", mean_squared_error(y_test, y_pred_poly))
print("MAE:", mean_absolute_error(y_test, y_pred_poly))
print("R2 Score:", r2_score(y_test, y_pred_poly))


# 15. CONCLUSION

print("\nConclusion:")
print("Thus, we have studied Regression Techniques and")
print("implemented Linear Regression and Polynomial Regression.")
