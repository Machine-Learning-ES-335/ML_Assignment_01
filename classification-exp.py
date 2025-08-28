import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold

np.random.seed(42)

# ------------------ Generate Dataset ------------------
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=2, class_sep=0.5
)

# Dataset
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Generated Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()     

# ------------------ Q2 a) Train-Test Split ------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)     #Split: 70% train, 30% test

# Convert y to pandas Series because: DecisionTree implementation requires y to be a pandas
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

# Initialize Decision Tree:
dt = DecisionTree(criterion='information_gain')  # or 'gini_index'


dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

acc = accuracy(y_test, y_pred)
classes = np.unique(y_test)
prec = [precision(y_test, y_pred, c) for c in classes]
rec = [recall(y_test, y_pred, c) for c in classes]

print("Q2 a) Decision Tree on Train-Test Split")
print()
print("____information_gain____")
print(f"Accuracy: {acc:.4f}")
print(f"Per-class Precision: {prec}")
print(f"Per-class Recall: {rec}")

print("\n***********************************************\n")

dt = DecisionTree(criterion='gini_index') 

dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

acc = accuracy(y_test, y_pred)
classes = np.unique(y_test)
prec = [precision(y_test, y_pred, c) for c in classes]
rec = [recall(y_test, y_pred, c) for c in classes]

print("____gini_index____")
print(f"Accuracy: {acc:.4f}")
print(f"Per-class Precision: {prec}")
print(f"Per-class Recall: {rec}")

print("\n***********************************************\n")

# ------------------ Q2 b) Nested Cross-Validation ------------------

best_depth = None
best_score = 0
best_criteria = None

kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'max_depth': list(range(1, 11))}

for depth in param_grid['max_depth']:
    for criterion in ["information_gain", "gini_index"]:
        fold_scores = []
        for train_idx, val_idx in kf.split(X):
            X = pd.DataFrame(X)
            y = pd.Series(y)
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Directly use DecisionTree without wrapper
            tree = DecisionTree(criterion=criterion, max_depth=depth)
            tree.fit(X_train_fold, y_train_fold)
            y_val_pred = tree.predict(X_val_fold)
            
            fold_scores.append(accuracy(y_val_fold, y_val_pred))

        mean_score = np.mean(fold_scores)
        print(f"Depth {depth}, Criteria {criterion[0]}: CV Accuracy = {mean_score:.4f}")

        if mean_score >= best_score:
            best_score = mean_score
            best_depth = depth
            best_criteria = criterion

print("\nQ2 b) Nested CV Result")
print(f"Optimal Depth: {best_depth}")
print(f"Best CV Accuracy: {best_score:.4f}")
print("Best criteria:", best_criteria)

