import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from typing import Union


# Custom type used for typing of the models used
Model = Union[
    LogisticRegression, 
    DecisionTreeClassifier, 
    RandomForestClassifier
]


def tune_hyperparameters(model: Model, model_name: str, train: tuple) -> Model:
    """
    Given an untrained model and the training data, tunes the hyperparameters 
    appropriate to the model type, then returns the tuned model and
    hyperparameters in a tuple.

    Args:
        model (Model): untrained model
        model_name (str): name of model
        train (tuple): contains two pd.DataFrame with features and labels

    Returns:
        Model: tuned model
    """
    X_train, y_train = train

    if isinstance(model, LogisticRegression):
        param_grid: dict = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "max_iter": [10000, 100000]
        }

    elif isinstance(model, DecisionTreeClassifier):
        param_grid: dict = {
            "max_depth": [None, 3, 6, 12, 24, 30],
            "min_samples_split": [2, 4, 6, 8, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5]
        }

    elif isinstance(model, RandomForestClassifier):
        param_grid: dict = {
            "max_depth": [None, 3, 6, 12, 24, 30],
            "min_samples_split": [2, 4, 6, 8, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "n_estimators": [50, 100, 200, 500]
        }

    grid_search: GridSearchCV = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=5, scoring="f1"
    )

    grid_search.fit(X_train, y_train)
    model, params = grid_search.best_estimator_, grid_search.best_params_

    print(f"\t- Model: {model_name}")
    
    for param in params:
        print(f"\t\t- {param}: {params[param]}")

    return model


def train_model(model: Model, model_name: str, train: tuple) -> Model:
    """
    Given an untrained model and training data, trains the model on the data.

    Args:
        model (Model): untrained model
        model_name (str): name of model
        train (tuple): contains two pd.DataFrame with features and labels

    Returns:
        model (Model): trained model
    """
    X_train, y_train = train

    # Trains model with training data
    model.fit(X_train, y_train)

    return model


def predict_income(model: Model, data: tuple) -> np.ndarray:
    """
    Given a trained model and test data, predicts a binary classification, then
    returns the predicted classifications.

    Args:
        model: trained model
        data (tuple): contains two pd.DataFrame with features and labels

    Returns:
        np.ndarray: contains binary classifications, predicted by the model
    """
    X, _ = data
    predictions: np.ndarray = model.predict_proba(X)
    predictions = np.where(predictions[:, 0] > predictions[:, 1], 0, 1)

    return predictions.reshape(-1, 1)


def store_predictions(
    predictions: np.ndarray, 
    data: tuple,
    output_path: Path
) -> None:
    """
    Given the model's predictions and the destination pathway, joins the
    model's predictions with the associated feature vector, then saves the
    new dataframe to a csv file.

    Args:
        predictions (np.ndarray): model output
        data (tuple): contains the feature vector
        output_path (Path): file path where predictions are saved
    """
    print(f"\t- Saving predictions...")
    X, _ = data
    predictions = pd.Series(
        np.array(['>50K' if pred == 1 else '<=50K' for pred in predictions]),
        name="income"
    )
    X.reset_index(drop=True, inplace=True)
    predictions.reset_index(drop=True, inplace=True)
    dataframe: pd.DataFrame = pd.concat([X, predictions], axis=1)
    dataframe.to_csv(output_path, index=False)
