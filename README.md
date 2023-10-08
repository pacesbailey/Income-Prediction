# Project 1: Income Prediction

This is an implemented solution to the prompt provided for a final project for the Intelligent Data Analysis and Machine Learning I class at Universität Potsdam. 

The prompt establishes the following problem setting: a polling institute seeks to estimate income groups given personal data, for which goal incomplete survey data of 30,000 individuals was provided (stored as `einkommen.train` in the `data` subfolder).

## Usage
This program is intended to be run using `main.py`, where each of the other py files are called. When running `main.py`, the following processes are undertaken:

- Data preprocessing (`dataloader.py` and `preprocessing.py`):

    - **Data loading**: The data set is read from the `einkommen.train` csv file.

    - **Handling missing values**: Missing values, denoted by "`?`" are replaced with "`np.nan`".

    - **Data exploration**: The data set is visualized, where numeric values are shown using histograms and categorical values are shown using bar plots.

    - **Feature normalization**: Numeric values are normalized in their respective ranges using log normalization.

    - **Categorical formatting**: Categorical data, save for the target attribute, are formatted and replaced with one-hot encodings. 

    - **Train-test split**: The training data is separated from the test data.

- Hyperparameter tuning (`model.py`):

    - Three models (**Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier**) undergo hyperparameter training using the **Grid Search** algorithm.

    - The optimal hyperparameters for each model are reported and the tuned model is fed into the next stage.

- Model evaluation (`evaluation.py`):

    - The tuned models are evaluated using **K-Fold Cross Validation**, reported the mean accuracy, recall, and F1 values for each model.

- Model training (`model.py`):

    - The tuned models are trained on the training data.

- Model prediction (`model.py`):

    - The models make predictions on the test data, then the predictions are stored in respective csv files and saved in the `data` subfolder.

## Experiments

- **Feature normalization**: Numeric features are normalized using log normalization to smooth the influence of values with large absolute values. There was a signification increase in the performance of the Logistic Regression classifier, both in terms of accuracy and F1 score. The other two models saw negligible benefits at evaluation time.

- **Feature exclusion**: Three features were selected for possible elimination during the training process: `birth_country`, `financial_gain`, and `financial_loss`. Each of these features showed extreme tendencies towards one value (`United-States`, `0`, and `0`, respectively). Eliminating all three resulted in a breakdown of the Logistic Regression classifier, while eliminating only `birth_country` resulted in an increase in performance, mostly for the Logisitic Regression classifier.

- **Hyperparameter tuning**: Multiple sets of configurations were considered for each model, ultimately resulting in the possible hyperparameters as detailed in `model.py`. 

- **Model comparison**: Through all experiments, the Random Forest classifier yielded the best results, sometimes far outperforming the other two and at other times performing only marginally better.

## Results

The final configuration which resulted in the included prediction files (saved as `predictions-logistic_regression.csv`, `predictions-decision_tree_classifier.csv`, and `predictions-random_forest_classifier.csv` in the `data` subfolder) yielded the results detailed in the table below. It is clear that the Random Forest Classifier only slightly outperforms the other two methods, where the Logistic Regression classifier slightly outperforms the Decision Tree Classifier in terms of accuracy and F1 score. The Decision Tree Classifier yields a higher recall score than the other two models.

|                          | Accuracy | Recall | F1     |
|--------------------------|----------|--------|--------|
| Logistic Regression      | 0.8382   | 0.5937 | 0.6436 |
| Decision Tree Classifier | 0.8154   | 0.6201 | 0.6267 |
| Random Forest Classifier | 0.8490   | 0.5787 | 0.6554 |# Income-Prediction
