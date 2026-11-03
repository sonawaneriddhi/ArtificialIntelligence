
# ==========================================================
# AIML GROUP C - ASSIGNMENT 1
# CLUSTERING TECHNIQUES
# K-MEANS AND HIERARCHICAL CLUSTERING
# ==========================================================

# STEP 1: IMPORTING REQUIRED LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==========================================================
# STEP 2: LOADING THE DATASET
# ==========================================================

df = pd.read_csv("Mall_Customers.csv")

# Reading first rows
print("\nFirst 5 Rows:")
print(df.head())

# Checking dimensions
print("\nShape of Dataset:")
print(df.shape)

# Checking data types
print("\nData Types:")
print(df.dtypes)

# Renaming column
df.rename(columns={'Genre': 'Gender'}, inplace=True)

# Printing information
print("\nDataset Information:")
df.info()

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Checking missing values
print("\nMissing Values:")
print(df.isna().sum())

# Checking total missing values
print("\nTotal Missing Values:")
print(df.isna().sum().sum())

# Checking duplicates
print("\nDuplicate Values:")
print(df.duplicated().sum())

# Checking unique values
print("\nUnique Values:")
print(df.nunique())

# Dropping CustomerID column
df.drop(columns=['CustomerID'], inplace=True)

print("\nDataset After Dropping CustomerID:")
print(df.head())

# ==========================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

# Checking outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=df.select_dtypes(include=np.number))
plt.title("Checking Outliers")
plt.show()

# Distribution of Annual Income
plt.figure(figsize=(7, 5))
sns.histplot(df['Annual Income (k$)'], kde=True)
plt.title("Distribution of Annual Income")
plt.show()

# Distribution of Age
plt.figure(figsize=(7, 5))
sns.histplot(df['Age'], kde=True)
plt.title("Distribution of Age")
plt.show()

# Distribution of Spending Score
plt.figure(figsize=(7, 5))
sns.histplot(df['Spending Score (1-100)'], kde=True)
plt.title("Distribution of Spending Score")
plt.show()

# Distribution of Gender
plt.figure(figsize=(7, 5))
sns.countplot(x='Gender', data=df)
plt.title("Distribution of Gender")
plt.show()

# Relationship between Annual Income, Spending Score and Gender
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Gender'
)
plt.title("Annual Income vs Spending Score")
plt.show()

# ==========================================================
# STEP 4: PREPROCESSING THE DATA BY SCALING FEATURES
# ==========================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

print("\nScaled Dataset:")
print(df.head(3))

# ==========================================================
# STEP 5: CHECKING CORRELATION AMONG FEATURES
# ==========================================================

print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ==========================================================
# STEP 6: APPLYING LABEL ENCODER
# ==========================================================

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])

print("\nDataset After Label Encoding:")
print(df.head(3))

# ==========================================================
# STEP 7: FINDING OPTIMAL NUMBER OF CLUSTERS
# USING ELBOW METHOD
# ==========================================================

from sklearn.cluster import KMeans

wcss_list = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df)
    wcss_list.append(kmeans.inertia_)

print("\nWCSS Values:")
print(wcss_list)

# Plotting Elbow Method
plt.figure(figsize=(8, 5))
sns.lineplot(x=range(2, 11), y=wcss_list, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# ==========================================================
# STEP 8: SILHOUETTE SCORE
# ==========================================================

from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df)

    labels = kmeans.predict(df)

    # Calculate Silhouette Score
    score = silhouette_score(df, labels)
    silhouette_scores.append(score)

print("\nSilhouette Scores:")
print(silhouette_scores)

# Plotting Silhouette Score
plt.figure(figsize=(8, 5))
sns.lineplot(
    x=range(2, 11),
    y=silhouette_scores,
    marker='o'
)
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

# ==========================================================
# STEP 9: APPLYING K-MEANS CLUSTERING ALGORITHM
# ==========================================================

num_clusters = 5

kmeans = KMeans(
    n_clusters=num_clusters,
    random_state=42,
    n_init=10
)

df['cluster'] = kmeans.fit_predict(df)

print("\nK-Means Clustering Results:")
print(df.head(3))

# ==========================================================
# STEP 10: VISUALIZING RESULTING CLUSTERS
# ==========================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='cluster',
    data=df,
    palette='viridis'
)

plt.title("K-Means Clustering")
plt.show()

# ==========================================================
# STEP 11: APPLYING HIERARCHICAL CLUSTERING
# ==========================================================

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Use original features without the K-Means cluster label
X = df.drop(columns=['cluster'])

# Compute Linkage Matrix
matrix = linkage(X, method='ward')

# Plot the Dendrogram
plt.figure(figsize=(10, 7))

dendrogram(
    matrix,
    orientation='top',
    distance_sort='descending',
    show_leaf_counts=True
)

plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# ==========================================================
# STEP 12: APPLY AGGLOMERATIVE CLUSTERING
# ==========================================================

# Set the number of clusters
num_clusters = 4

agg_clustering = AgglomerativeClustering(
    n_clusters=num_clusters,
    linkage='ward'
)

clusters = agg_clustering.fit_predict(X)

# Add cluster labels to original data
df['Cluster1'] = clusters

print("\nHierarchical Clustering Results:")
print(df.head(3))

# ==========================================================
# STEP 13: VISUALIZING HIERARCHICAL CLUSTERS
# ==========================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster1',
    palette='viridis'
)

plt.title("Hierarchical Clustering")
plt.show()

# ==========================================================
# CONCLUSION
# ==========================================================

print("\nConclusion:")
print(
    "The clustering analysis successfully segmented customers "
    "into distinct groups, revealing valuable insights into "
    "their spending behavior."
)

print("\nClustering Assignment Completed Successfully!")
