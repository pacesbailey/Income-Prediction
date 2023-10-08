import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from dataloader import DataLoader
from evaluation import cross_validate
from model import predict_income, store_predictions, train_model, \
    tune_hyperparameters
from preprocessing import preprocess_data, split_data


# Relevant file paths
directory: Path = Path(__file__).resolve().parent
data_directory: Path = directory / "data"
data_path: Path = data_directory / "einkommen.train"


def main() -> None:
    """
    Given a data set with partially incomplete data concerning income, this
    program creates, trains, evaluates, and compares three different approaches
    to predicting the missing values. The models under comparison are Logistic
    Regression, Decision Tree Classifier, and Random Forest Classifier. This
    is carried out in four steps: (1) data preprocessing, (2) hyperparameter
    tuning, (3) model training, and (4) model evaluation. 
    """
    models: list = [
        [LogisticRegression(), "Logistic Regression"],
        [DecisionTreeClassifier(), "Decision Tree Classifier"],
        [RandomForestClassifier(), "Random Forest Classifier"]
    ]

    # 1. Data preprocessing
    print("\n--------- DATA PREPROCESSING ---------")
    dataloader: DataLoader = DataLoader(data_path)
    dataframe: pd.DataFrame = preprocess_data(dataloader)
    print("\t- Splitting data set into training and test subsets...")
    train, test = split_data(dataframe)

    # 2. Hyperparameter tuning
    print("\n--------- HYPERPARAMETER TUNING ---------")
    for model in models:
        model[0] = tune_hyperparameters(model[0], model[1], train)

    # 3. Model evaluation
    model_scores: dict = {}
    print("\n--------- MODEL EVALUATION ---------")
    for model in models:
        # loss: float = compute_loss(model[0], train)
        model_scores[model[1]] = cross_validate(model[0], model[1], train, 5)

    print("\t\t\t\t\tAccuracy  Recall    F1")
    for model in model_scores:
        accuracy: float = model_scores[model]["accuracy"][1]
        recall: float = model_scores[model]["recall"][1]
        f1: float = model_scores[model]["f1"][1]

        if len(model) < 24:
            print(f"\t- {model}\t\t{accuracy}    {recall}    {f1}")
        else:
            print(f"\t- {model}\t{accuracy}    {recall}    {f1}")

    # 4. Model training
    print("\n--------- MODEL TRAINING ---------")
    print("\t- Training the tuned models...")
    for model in models:
        model[0] = train_model(model[0], model[1], train)

    # 5. Model prediction
    print("\n--------- MODEL PREDICTION ---------")
    for model in models:
        name: str = model[1].lower().replace(" ", "_")
        output_path: Path = data_directory / f"predictions-{name}.csv"
        
        dataloader.load_data()
        dataloader.handle_missing()
        _, features = split_data(dataloader.dataframe)

        predictions: np.ndarray = predict_income(model[0], test)
        store_predictions(predictions, features, output_path)
    
    print("\n")


if __name__ == "__main__":
    main()
