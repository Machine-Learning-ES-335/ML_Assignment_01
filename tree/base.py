
# The current code given is for the Assignment 1.
# You will be expected to use this to make trees for:
# > discrete input, discrete output
# > real input, real output
# > real input, discrete output
# > discrete input, real output

from dataclasses import dataclass
from typing import Literal, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

# new:
@dataclass
class Node:
    # Each node can have a left child, right child, a splitting feature,threshold, and leaf values
    feature: Union[str, None] = None
    threshold: Union[float, None] = None
    left: Union['Node', None] = None
    right: Union['Node', None] = None
    value: Union[float, int, None] = None  
    is_leaf: bool = False
# new end;


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int = 3  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=3):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.is_regression = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
       
        #Function to train and construct the decision tree
        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        X = one_hot_encoding(X)                                           #Convert discrete features to one-hot encoding
        self.is_regression = check_ifreal(y)                              #Check if output is real or discrete
        features = list(X.columns)
        self.root = self._build_tree(X, y, features, depth=0)


    #new:
    def _build_tree(self, X, y, features, depth):
        # Stopping conditions
        if len(y.unique()) == 1 or depth >= self.max_depth or len(features) == 0:
            leaf_value = y.mean() if self.is_regression else y.mode()[0]
            return Node(value=leaf_value, is_leaf=True)

        # Finding best feature to split
        best_feature, best_threshold = opt_split_attribute(X, y, self.criterion, features)
        if best_feature is None:
            leaf_value = y.mean() if self.is_regression else y.mode()[0]
            return Node(value=leaf_value, is_leaf=True)

        X_left, y_left, X_right, y_right = split_data(X, y, best_feature, best_threshold)
        if len(y_left) == 0 or len(y_right) == 0:
            leaf_value = y.mean() if self.is_regression else y.mode()[0]
            return Node(value=leaf_value, is_leaf=True)

        left_child = self._build_tree(X_left, y_left, features, depth + 1)
        right_child = self._build_tree(X_right, y_right, features, depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    # new end


    def predict(self, X: pd.DataFrame) -> pd.Series:
        
        # Funtion to run the decision tree on test inputs
        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        X = one_hot_encoding(X)
        preds = X.apply(lambda row: self._predict_row(row, self.root), axis=1)
        return pd.Series(preds)


    # new:
    def _predict_row(self, row, node):
        if node.is_leaf:
            return node.value
        # Decide which branch to take
        val = row[node.feature] if node.feature in row else 0  # handle missing one-hot columns
        if node.threshold is not None:
            if val <= node.threshold:
                return self._predict_row(row, node.left)
            else:
                return self._predict_row(row, node.right)
        else:
            # discrete: go left if value is 1 (one-hot), else right
            if val == 1:
                return self._predict_row(row, node.left)
            else:
                return self._predict_row(row, node.right)
    # new end

    def plot(self) -> None:
        
        # Function to plot the tree

        # Output Example:
        # ?(X1 > 4)
        #     Y: ?(X2 > 7)
        #         Y: Class A
        #         N: Class B
        #     N: Class C
        # Where Y => Yes and N => No

        def recurse(node, indent=""):
            if node.is_leaf:
                print(indent + "Leaf:", node.value)
            else:
                if node.threshold is not None:                                          #for continuous features
                    print(f"{indent}?({node.feature} <= {node.threshold})")
                else:
                    print(f"{indent}?({node.feature})")
                recurse(node.left, indent + "  Y-> ")
                recurse(node.right, indent + "  N-> ")
        recurse(self.root)

