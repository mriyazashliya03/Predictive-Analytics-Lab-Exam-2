import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ==========================================
# Task 1: Exploratory Data Analysis (EDA)  
# ==========================================
# Load the dataset
file_path = 'Lab_Exam_binary_classification_dataset.csv'
df = pd.read_csv(file_path)

print("--- Dataset Information ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())

# Check for missing values
print("\n--- Missing Values Check ---")
print(df.isnull().sum())

# Visualize distributions and feature relationships
sns.pairplot(df, hue=df.columns[-1], diag_kind='kde', palette='husl')
plt.suptitle("EDA: Feature Pairplot", y=1.02)
plt.show()

# ==========================================
# Task 2: Build Classification Model  
# ==========================================
# Assuming the last column is the target (y) and everything else is features (X)
X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values   

# Split the data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
print("\nModel trained successfully using Logistic Regression.")

# ==========================================
# Task 3: Plot Decision Boundary 
# ==========================================
def plot_decision_boundary(X, y, model):
    # Create a grid to plot in
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02  # step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict across the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdYlBu')
    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Visualize the boundary using the training data
plot_decision_boundary(X_train, y_train, model)

# ==========================================
# Task 4: Evaluate and Report Performance  
# ==========================================
y_pred = model.predict(X_test)

# Calculate specific metrics
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, average='binary')
rec = recall_score(y_test, y_pred, average='binary')
f1  = f1_score(y_test, y_pred, average='binary')

print("\n--- Model Performance Report ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\n--- Confusion Matrix ---")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("\n--- Full Classification Report ---")
print(classification_report(y_test, y_pred))
