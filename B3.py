
# ==========================================================
# AIML LAB - ASSIGNMENT 3
# CLASSIFICATION USING DECISION TREE AND NAIVE BAYES
# Dataset: Titanic-Dataset.csv
# ==========================================================

# STEP 1: IMPORTING REQUIRED LIBRARIES

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    fbeta_score
)

# ==========================================================
# STEP 2: LOADING THE DATASET
# ==========================================================

titanic = pd.read_csv("MarvellousTitanicDataset.csv")

print("First 5 rows:")
print(titanic.head())

# ==========================================================
# STEP 3: DATA PREPARATION
# ==========================================================

print("\nFirst 5 rows:")
print(titanic.head())

print("\nLast 5 rows:")
print(titanic.tail())

print("\nDimensions of dataset:")
print(titanic.shape)

print("\nDataset Information:")
titanic.info()

print("\nDescriptive Statistics:")
print(titanic.describe())

print("\nMissing Values:")
print(titanic.isnull().sum())

print("\nDuplicate Records:")
print(titanic.duplicated().sum())

# ==========================================================
# STEP 4: DATA PREPROCESSING
# ==========================================================

# Handling missing values

titanic['Age'] = titanic['Age'].fillna(
    titanic['Age'].mean()
)

titanic['Embarked'] = titanic['Embarked'].fillna(
    titanic['Embarked'].mode()[0]
)

titanic['Fare'] = titanic['Fare'].fillna(
    titanic['Fare'].mean()
)

# Checking unique values

print("\nUnique values in each column:")
print(titanic.nunique())

# Dropping unnecessary columns

titanic.drop(
    columns=['PassengerId', 'Name', 'Ticket', 'Cabin'],
    inplace=True
)

# Converting categorical variables into numerical values

le = LabelEncoder()

titanic['Sex'] = le.fit_transform(titanic['Sex'])

titanic['Embarked'] = le.fit_transform(
    titanic['Embarked']
)

print("\nProcessed Dataset:")
print(titanic.head())

# ==========================================================
# STEP 5: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

# Check how many passengers survived

sns.countplot(x=titanic['Survived'])
plt.title("Number of Passengers Survived")
plt.show()

# Compare survival by gender

sns.countplot(
    x=titanic['Survived'],
    hue=titanic['Sex']
)
plt.title("Survival by Gender")
plt.show()

# Compare survival by passenger class

sns.countplot(
    x=titanic['Pclass'],
    hue=titanic['Survived']
)
plt.title("Survival by Passenger Class")
plt.show()

# Analyze relationship between age and survival

sns.barplot(
    x=titanic['Survived'],
    y=titanic['Age']
)
plt.title("Age and Survival")
plt.show()

# Analyze fare and survival

sns.barplot(
    x=titanic['Survived'],
    y=titanic['Fare']
)
plt.title("Fare and Survival")
plt.show()

# Survival by gender and passenger class

sns.barplot(
    x=titanic['Pclass'],
    y=titanic['Survived'],
    hue=titanic['Sex']
)
plt.title("Survival by Gender and Passenger Class")
plt.show()

sns.catplot(
    x="Pclass",
    hue="Survived",
    col="Sex",
    data=titanic,
    kind="count"
)
plt.show()

# Survival by embarkation port

sns.countplot(
    x=titanic['Survived'],
    hue=titanic['Embarked']
)
plt.title("Survival by Embarkation Port")
plt.show()

# ==========================================================
# STEP 6: EXTRACTING FEATURES AND TARGET
# ==========================================================

X = titanic.drop("Survived", axis=1)

y = titanic['Survived']

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# ==========================================================
# STEP 7: TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# STEP 8: FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ==========================================================
# STEP 9: DECISION TREE CLASSIFIER
# ==========================================================

# Build the classification model

dt = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=42
)

dt.fit(X_train, y_train)

# Make predictions

y_pred_dt = dt.predict(X_test)

# Evaluate the model

print("\n========== DECISION TREE RESULTS ==========")

print(
    "Decision Tree Accuracy:",
    accuracy_score(y_test, y_pred_dt)
)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# Plot the decision tree

plt.figure(figsize=(16, 10))

plot_tree(
    dt,
    feature_names=X.columns,
    class_names=['Not Survived', 'Survived'],
    filled=True
)

plt.title("Decision Tree Classifier")
plt.show()

# ==========================================================
# STEP 10: NAIVE BAYES CLASSIFIER
# ==========================================================

# Build the classification model

nb = GaussianNB()

nb.fit(X_train, y_train)

# Make predictions

y_pred_nb = nb.predict(X_test)

# Evaluate the model

print("\n========== NAIVE BAYES RESULTS ==========")

print(
    "Naive Bayes Accuracy:",
    accuracy_score(y_test, y_pred_nb)
)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

# ==========================================================
# STEP 11: HANDLING IMBALANCED DATA USING SMOTE
# ==========================================================

from imblearn.over_sampling import SMOTE

# Apply SMOTE only to training data

oversample = SMOTE(random_state=42)

X_sampled, y_sampled = oversample.fit_resample(
    X_train,
    y_train
)

print("\nClass distribution after SMOTE:")
print(y_sampled.value_counts())

# Visualize balanced dataset

sns.countplot(x=y_sampled)
plt.title("Class Distribution After SMOTE")
plt.show()

# ==========================================================
# STEP 12: DECISION TREE USING SMOTE DATA
# ==========================================================

# Create Decision Tree classifier

classifier = DecisionTreeClassifier(
    random_state=42
)

# Fit the model

classifier.fit(X_sampled, y_sampled)

# Predict the test set result

y_pred = classifier.predict(X_test)

# ==========================================================
# STEP 13: EVALUATE THE MODEL
# ==========================================================

print("\n========== SMOTE DECISION TREE RESULTS ==========")

print(
    "Confusion Matrix:",
    confusion_matrix(y_test, y_pred)
)

print(
    "Accuracy Score:",
    accuracy_score(y_test, y_pred)
)

print(
    "Precision Score:",
    precision_score(y_test, y_pred)
)

print(
    "Recall Score:",
    recall_score(y_test, y_pred)
)

print(
    "F-beta Score:",
    fbeta_score(y_test, y_pred, beta=1)
)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================================
# STEP 14: VISUALIZE DECISION TREE
# ==========================================================

plt.figure(figsize=(20, 10))

tree = plot_tree(
    decision_tree=classifier,
    feature_names=X.columns,
    class_names=['0', '1'],
    filled=True
)

plt.title("Decision Tree After SMOTE")

plt.savefig("Decision Tree.png")

plt.show()

# ==========================================================
# STEP 15: MODEL COMPARISON
# ==========================================================

print("\n========== MODEL COMPARISON ==========")

print(
    "Decision Tree Accuracy:",
    accuracy_score(y_test, y_pred_dt)
)

print(
    "Naive Bayes Accuracy:",
    accuracy_score(y_test, y_pred_nb)
)

print(
    "SMOTE Decision Tree Accuracy:",
    accuracy_score(y_test, y_pred)
)

print("\n========== PROGRAM COMPLETED ==========")
