import pandas as pd

from dataloader import DataLoader


def preprocess_data(dataloader: DataLoader) -> pd.DataFrame:
    """
    Loads data from a csv file into a dataframe using a dataloader object, 
    processes missing values by replacing non-target values with the mode of
    their respective features and placing np.nan into the missing target
    values, then transforms categorical values into one-hot encodings, finally
    returning a pd.DataFrame.

    Args:
        dataloader (DataLoader): unmanipulated DataLoader object

    Returns:
        pd.DataFrame: manipulated pd.DataFrame from csv file
    """        
    print("\t- Loading data set...")
    dataloader.load_data()
    print("\t- Handling missing values...")
    dataloader.handle_missing()
    print("\t- Exploring data set...")
    dataloader.explore_data_set()
    print("\t- Excluding features...")
    dataloader.exclude_features()
    print("\t- Normalizing numeric values...")
    dataloader.normalize_numeric()
    print("\t- Formatting categorical values...")
    dataloader.format_categorical()
    
    return dataloader.dataframe


def split_data(dataframe: pd.DataFrame) -> tuple[tuple, tuple]:
    """
    Splits the formatted dataframe into training and test subsets.

    Args:
        dataframe (pd.DataFrame): formatted dataframe

    Returns:
        tuple[tuple, tuple]: contains tuples which contain data subsets
    """
    # Splits the dataframe into train and test dataframes
    train: pd.DataFrame = dataframe[dataframe['income'].notna()]
    test: pd.DataFrame = dataframe[dataframe['income'].isna()]

    # Splits the train and test dataframes into dataframes containing only
    # features and labels respectively
    X_train: pd.DataFrame = train.drop('income', axis=1)
    y_train: pd.DataFrame = train['income']
    X_test: pd.DataFrame = test.drop('income', axis=1)
    y_test: pd.DataFrame = test['income']

    # Stores the X and y values into tuples for convenience
    train: tuple = X_train, y_train
    test: tuple = X_test, y_test
    
    return train, test
