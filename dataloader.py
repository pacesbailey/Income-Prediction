import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path


class DataLoader:
    """
    Class used to load data from a csv file into a pd.DataFrame, perform 
    various transformations to it in preparation for machine learning models,
    and return the formatted or unformatted dataframe.
    """

    def __init__(self, file_path: Path) -> None:
        self.file_path: Path = file_path
        self.dataframe: pd.DataFrame | None = None

    def load_data(self) -> None:
        """
        Loads the dataset from a csv file into a pd.DataFrame then adds column
        headers as specified in to project prompt.
        """
        column_headers: list = [
            'age', 'employment_type', 'selection_bias_weight', 
            'education_level', 'schooling_period', 'marital_status', 
            'employment_area', 'partnership', 'ethnicity', 'gender', 
            'financial_gains', 'financial_losses', 'weekly_working_time', 
            'birth_country', 'income'
        ]
        
        self.dataframe = pd.read_csv(self.file_path, header=None)
        self.dataframe.columns = column_headers

    def handle_missing(self) -> None:
        """
        Replaces missing values with either the mode of their respective 
        features or a np.nan value depending on if they are a feature for
        analysis or the target attribute.
        """
        # Replaces missing values with np.nan
        self.dataframe.replace(' ?', np.nan, inplace=True)

        # Separates the target attribute to maintain its missing values
        # Replaces missing values of the rest with their respective modes
        missing_values: pd.DataFrame = self.dataframe.drop('income', axis=1)
        missing_values.fillna(missing_values.mode().iloc[0], inplace=True)

        missing_values['income'] = self.dataframe['income']
        self.dataframe = missing_values
    
    def explore_data_set(self) -> None:
        """
        Through data visualization, explores the makeup of all of the
        attributes, save for the target attribute ("income").
        """
        # Separates numeric and categorical columns for analysis
        numeric: pd.DataFrame = self.dataframe.select_dtypes(
            include=["int64", "float64"]
        )
        categorical: pd.DataFrame = self.dataframe.select_dtypes(
            include=["object"]
        )
        
        # Plots numeric features
        fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
        fig1.suptitle("Numeric Features (1/2)")
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        fig2.suptitle("Numeric Features (2/2)")

        axes1 = axes1.flatten()
        axes2 = axes2.flatten()
        
        for idx, col in enumerate(numeric.columns[:3]):
            axes1[idx].hist(numeric[col], bins=10)
            axes1[idx].set_title(col.replace("_", " ").capitalize())

        for idx, col in enumerate(numeric.columns[3:]):
            axes2[idx].hist(numeric[col], bins=10)
            axes2[idx].set_title(col.replace("_", " ").capitalize())

        # Plots categorical features
        fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
        fig3.suptitle("Categorical Features (1/3)")
        fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))
        fig4.suptitle("Categorical Features (2/3)")
        fig5, axes5 = plt.subplots(1, 3, figsize=(15, 5))
        fig5.suptitle("Categorical Features (3/3)")

        axes3 = axes3.flatten()
        axes4 = axes4.flatten()
        axes5 = axes5.flatten()
        
        for idx, col in enumerate(categorical.columns[:3]):
            counts = categorical[col].value_counts()
            axes3[idx].bar(counts.index, counts.values)
            axes3[idx].set_title(col.replace("_", " ").capitalize())
            axes3[idx].tick_params(axis="x", rotation=45)

        for idx, col in enumerate(categorical.columns[3:6]):
            counts = categorical[col].value_counts()
            axes4[idx].bar(counts.index, counts.values)
            axes4[idx].set_title(col.replace("_", " ").capitalize())
            axes4[idx].tick_params(axis="x", rotation=45)

        for idx, col in enumerate(categorical.columns[6:]):
            counts = categorical[col].value_counts()
            axes5[idx].bar(counts.index, counts.values)
            axes5[idx].set_title(col.replace("_", " ").capitalize())
            axes5[idx].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def exclude_features(self) -> None:
        """
        Excludes very imbalanced features from the training procedure.
        """
        # features: list = ["birth_country", "financial_gains", "financial_losses"]
        features: list = ["birth_country"]
        self.dataframe.drop(columns=features, inplace=True)

    def normalize_numeric(self) -> None:
        """
        Normalizes numeric data using log to smooth the influence of large 
        absolute values.
        """
        numeric: pd.DataFrame = self.dataframe.select_dtypes(
            include=["int64", "float64"]
        )
        columns: list = [col for col in numeric.columns]
        self.dataframe[columns] = np.log1p(self.dataframe[columns])

    def format_categorical(self) -> None:
        """
        Identifies the categorical features, as specified in the prompt,
        creates one-hot encodings, then replaces the target binary values
        with either 0 or 1 for ease of use.
        """
        categorical: pd.DataFrame = self.dataframe.select_dtypes(include=["object"])
        columns = [col for col in categorical.columns if col != "income"]
        self.dataframe = pd.get_dummies(self.dataframe, columns=columns)
        self.dataframe['income'].replace({' <=50K': 0, ' >50K': 1}, inplace=True)
