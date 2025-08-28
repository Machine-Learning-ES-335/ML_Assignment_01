"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    
    # If X is numeric numpy array
    if isinstance(X, np.ndarray):
        return pd.DataFrame(X)
    # Otherwise apply one-hot
    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    
    # Checks if the given series has real values or discrete values
    if pd.api.types.is_categorical_dtype(y):
        return False
    elif pd.api.types.is_numeric_dtype(y):                            #If y is numeric: int or float
        return np.issubdtype(y.dtype, np.floating)
    else:
        raise ValueError(f"Unsupported dtype: {y.dtype}")


def entropy(Y: pd.Series) -> float:
    
    # Entropy for a discrete target Y
    probs = Y.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-9)) 


def gini_index(Y: pd.Series) -> float:
    
    # Gini index for a discrete target Y
    probs = Y.value_counts(normalize=True)
    return 1 - np.sum(probs ** 2)

def mse(Y:pd.Series) -> float:

    # Mean squared error for a real-valued target
    return ((Y-Y.mean()) ** 2).mean()


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    
    #Calculate information gain for a split on attribute `attr` with target Y
    # criterion: "information_gain" (entropy), "gini_index", or "mse"

   # If Y is real and criterion is :information_gain, we use MSE instead
    if check_ifreal(Y):
        parent_score = mse(Y)
    else:
        if criterion == "information_gain":
            parent_score = entropy(Y)
        elif criterion == "gini_index":
            parent_score = gini_index(Y)
        else:
            raise ValueError("Unknown criterion for discrete output")
    
    # For continuous features spliting at median
    if np.issubdtype(attr.dtype, np.number):
        threshold = attr.median()
        left = Y[attr <= threshold]
        right = Y[attr > threshold]
    else:
        # discrete features
        left = Y[attr == attr.mode()[0]]
        right = Y[attr != attr.mode()[0]]
    
    # Weighted score
    n = len(Y)
    n_left = len(left)
    n_right = len(right)
    
    if check_ifreal(Y):
        score = n_left/n * mse(left) + n_right/n * mse(right)
    else:
        if criterion == "information_gain":
            score = n_left/n * entropy(left) + n_right/n * entropy(right)
        elif criterion == "gini_index":
            score = n_left/n * gini_index(left) + n_right/n * gini_index(right)
    
    return parent_score - score 


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, features: list):
   
    #Find the attribute and threshold (if real) that gives max info gain
    # Returns: best_feature, best_threshold (None for discrete)
    # Using feature:list instead of pd.series: avoids unnecessary pandas operations and makes the code cleaner and faster when iterating.

    best_gain = -np.inf
    best_feature = None
    best_threshold = None
    
    for feature in features:
        col = X[feature]
        # If continuous, try median split
        if np.issubdtype(col.dtype, np.number):
            threshold = col.median()
            left = y[col <= threshold]
            right = y[col > threshold]
            if len(left) == 0 or len(right) == 0:
                continue
            gain = information_gain(y, col, criterion)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
        else:
            # discrete
            gain = information_gain(y, col, criterion)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = None
    return best_feature, best_threshold


def split_data(X: pd.DataFrame, y: pd.Series, attribute, threshold=None):
   
    # Split dataset based on attribute.
    # - For continuous features, use threshold
    # - For discrete features, split on equality with mode (most common value)
    # Returns: X_left, y_left, X_right, y_right

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    col = X[attribute]
    if threshold is not None:
        mask = col <= threshold
    else:
        mode_val = col.mode()[0]
        mask = col == mode_val
    
    X_left = X[mask].reset_index(drop=True)
    y_left = y[mask].reset_index(drop=True)
    
    X_right = X[~mask].reset_index(drop=True)
    y_right = y[~mask].reset_index(drop=True)
    
    return X_left, y_left, X_right, y_right
