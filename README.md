import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("C:\\Users\\hp\\Downloads\\archive (15)\\WineQT.csv")  # Change file path if necessary

# Display first few rows
print(df.head())

# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Data visualization
sns.histplot(df["quality"], bins=6, kde=True)
plt.title("Wine Quality Distribution")
plt.show()

# Define features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize classifiers
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SGD Classifier": SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
    "SVC": SVC(kernel='linear', random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
