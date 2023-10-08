import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from model import Model


def cross_validate(
    model: Model, 
    model_name: str, 
    train: tuple, 
    n_splits: int
) -> dict:
    """
    Cross-validates the performance of the trained model using the training
    data. Iteratively separates the training data into k folds or parts, using
    one of them as a validation subset. In each iteration, its performance is
    scored according to the specified metric. At the end, the scores, their
    mean value, and the standard deviation are given.

    Args:
        model: trained model
        model_name (str): name of model
        train (tuple): contains two pd.DataFrame with features and labels
        n_splits (int): number of folds for cross-validation

    Returns:
        dict: contains the name of the score, the scores, the mean value, and 
              the standard deviation
    """
    X_train, y_train = train
    kfold: KFold = KFold(n_splits=n_splits, shuffle=True)
    
    accuracy: np.ndarray = cross_val_score(
        model, 
        X_train, y_train, 
        scoring="accuracy", cv=kfold
    )
    recall: np.ndarray = cross_val_score(
        model,
        X_train, y_train,
        scoring="recall", cv=kfold
    )
    f1: np.ndarray = cross_val_score(
        model,
        X_train, y_train,
        scoring="f1", cv=kfold
    )
    
    scores: dict = {
        "accuracy": [
            accuracy.tolist(), 
            round(accuracy.mean(), 4), 
            accuracy.std()
        ],
        "recall": [
            recall.tolist(), 
            round(recall.mean(), 4), 
            recall.std()
        ],
        "f1": [
            f1.tolist(), 
            round(f1.mean(), 4), 
            f1.std()
        ]
    }

    return scores
