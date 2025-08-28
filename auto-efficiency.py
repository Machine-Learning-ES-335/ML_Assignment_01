
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor  # For comparison

np.random.seed(42)

# ------------------ Read and Clean Data ------------------
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Removing rows with missing horsepower
data = data[data["horsepower"] != '?']
data['horsepower'] = data['horsepower'].astype(float)

# Converting 'car_name' to integer labels:
data['car name'] = pd.factorize(data['car name'])[0]

# Features and target
X = data.drop(columns=["mpg"])
y = data["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

# ------------------ Q3 a) Custom Decision Tree ------------------
dt = DecisionTree(criterion='information_gain')
dt.fit(X_train, y_train)
y_pred_custom = dt.predict(X_test)

# Since it's regression, we can use RMSE or R2

rmse_custom = rmse(y_test, y_pred_custom)
mae_custom = mae(y_test, y_pred_custom)

print("Custom Decision Tree Performance:")
print(f"RMSE: {rmse_custom:.4f}")
print(f"MAE: {mae_custom:.4f}")


# ------------------ Q3 b) scikit-learn Decision Tree ------------------
sk_dt = DecisionTreeRegressor(criterion='friedman_mse', max_depth=None, random_state=42)
sk_dt.fit(X_train, y_train)
y_pred_sklearn = sk_dt.predict(X_test)

rmse_sklearn = rmse(y_test, y_pred_sklearn)
mae_sklearn = mae(y_test, y_pred_sklearn)

print("\nscikit-learn Decision Tree Performance:")
print(f"RMSE: {rmse_sklearn:.4f}")
print(f"MAE: {mae_sklearn:.4f}")

# Plot: 
plt.scatter(y_test, y_pred_custom, label='Custom DT', alpha=0.7)
plt.scatter(y_test, y_pred_sklearn, label='scikit-learn DT', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Ideal')
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Decision Tree Regression: Custom vs scikit-learn")
plt.legend()
plt.show()

