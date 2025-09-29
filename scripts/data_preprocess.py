import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Paths
DATA_PATH = "D:\\banglore internship\\major_project\\Early Flood Prediction System\\data\\flood.csv"
PROCESSED_DATA_PATH = "D:\\banglore internship\\major_project\\Early Flood Prediction System\\data\\flood_processed.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Load dataset
print("First 5 rows of the dataset : \n")
print(df.head())

# Dataset Information
print("Dataset Information : \n")
print(df.info())

# Dataset Statistics report
print("Dataset Statistics report : \n")
print(df.describe())

# Check missing values
print("Missing values per column: \n")
print(df.isnull().sum())

# Fill missing numeric values with column mean
df.fillna(df.mean(), inplace=True)

# Features and target
X = df.drop('FloodProbability', axis=1)
y = df['FloodProbability']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_rounded = X_scaled.round(3)

# Save processed data
df_scaled = pd.DataFrame(X_scaled_rounded, columns=X.columns)
df_scaled['FloodProbability'] = y.values
df_scaled.to_csv(PROCESSED_DATA_PATH, index=False)

print(f"Processed data saved at {PROCESSED_DATA_PATH}")
