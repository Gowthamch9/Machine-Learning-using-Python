# 🤖 Machine Learning using Python  

This repository contains a Jupyter Notebook (`machine learning.ipynb`) that demonstrates the application of **Supervised and Unsupervised Machine Learning algorithms** using Python.  

The goal is to explore how different ML models can be applied to real-world datasets for **classification, regression, and clustering tasks**.  

---

## 📂 Repository Structure  

- **`machine learning.ipynb`** → End-to-end notebook covering data preprocessing, feature engineering, model training, and evaluation.  

---

## 🛠️ Skills & Concepts Demonstrated  

- **Data Preprocessing**  
  - Handling missing values & outliers  
  - Feature scaling (StandardScaler)  
  - Balancing data using techniques like RandomOverSampler  

- **Supervised Learning Models**  
  - Logistic Regression  
  - Support Vector Machine (SVC)  
  - K-Nearest Neighbors (KNN)  
  - Gaussian Naive Bayes  
  - Random Forest Classifier  

- **Unsupervised Learning Models**  
  - K-Means Clustering  
  - Principal Component Analysis (PCA) for dimensionality reduction  
  - Neural Networks (basic implementation)  

- **Model Evaluation**  
  - Accuracy, Precision, Recall, F1 Score  
  - Confusion Matrix & ROC Curve  
  - Cross-validation for robust results  

---

## ⚡ Example Workflow  

```python
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("dataset.csv")

# Preprocess
X = df.drop("Target", axis=1)
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
