from typing import Union  #Union used to declare that a variable/parameter can be more than one type.
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
   
    # Accuracy = correct predictions / total predictions
    
    assert y_hat.size == y.size    #checks if sizes of y_hat and y are equal
    return (y_hat == y).sum() / len(y)



def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    
    # Precision = TP / (TP + FP)
    assert y_hat.size == y.size
    tp = ((y_hat == cls) & (y == cls)).sum()
    fp = ((y_hat == cls) & (y != cls)).sum()
    if tp + fp == 0:
        return 0.0
    return (tp / (tp + fp))


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    
    # Recall = TP / (TP + FN)
    assert y_hat.size == y.size
    tp = ((y_hat == cls) & (y == cls)).sum()
    fn = ((y_hat != cls) & (y == cls)).sum()
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    
    # Root mean squared error

    assert y_hat.size == y.size
    return np.sqrt(((y_hat - y) ** 2).mean())


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    
    # Mean absolute error

    assert y_hat.size == y.size
    return (y_hat - y).abs().mean()