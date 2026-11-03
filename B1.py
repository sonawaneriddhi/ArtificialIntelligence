
# TITANIC DATASET - DATA PREPROCESSING

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# 1. Load Dataset from CSV
titanic = pd.read_csv("MarvellousTitanicDataset.csv")

print("First 5 rows of dataset:")
print(titanic.head())


# 2. Data Cleaning
# Drop irrelevant columns
titanic = titanic.drop(['Cabin'], axis=1)


# 3. Missing Value Handling
# Fill numeric missing values with median
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

# Fill categorical missing values with mode
titanic['Embarked'] = titanic['Embarked'].fillna(
    titanic['Embarked'].mode()[0]
)


# 4. Feature Scaling
scaler = StandardScaler()

titanic['Fare_scaled'] = scaler.fit_transform(
    titanic[['Fare']]
)


# 5. Encoding Techniques
encoder = LabelEncoder()

titanic['Sex_encoded'] = encoder.fit_transform(titanic['Sex'])

titanic['Class_encoded'] = encoder.fit_transform(titanic['Pclass'])


# 6. Statistical Analysis
print("\nStatistical Analysis:")
print(titanic.describe())


# 7. Data Visualization

# Histogram of Age
sns.histplot(titanic["Age"])
plt.title("Age Distribution")
plt.show()

# Boxplot of Fare
sns.boxplot(x=titanic["Fare"])
plt.title("Fare Distribution")
plt.show()


# 8. Train-Test Split

X = titanic[['Age', 'Fare_scaled', 'Sex_encoded', 'Class_encoded']]

y = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 9. Display Train-Test Shapes
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nData Preprocessing Completed Successfully!")
