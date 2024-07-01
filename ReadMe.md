# **Wine Quality Prediction**

## Overview
https://prezi.com/view/cGMDbSDrSKbz3AmhDTUb/
This repository contains code for building and evaluating machine learning models to predict wine quality based on various chemical attributes. The models are developed using the scikit-learn library in Python.
**initial dataframe:**winequality-red.csv
****
## Libraries Used

- `pandas` - For handling data in DataFrame format.
- `numpy` - For numerical computations.
- `matplotlib.pyplot` - For creating visualizations.
- `seaborn` - For advanced visualizations.
- `train_test_split` from `sklearn.model_selection` - For splitting data into training and testing sets.
- `KNeighborsRegressor` from `sklearn.neighbors` - For K-Nearest Neighbors regression.
- `MinMaxScaler`, `StandardScaler` from `sklearn.preprocessing` - For data normalization or standardization.
- `GridSearchCV` from `sklearn.model_selection` - For hyperparameter tuning using grid search.
- `GradientBoostingRegressor` from `sklearn.ensemble` - For Gradient Boosting regression.
- `cross_val_score` from `sklearn.model_selection` - For performing cross-validation.

## Data Loading and Preprocessing

The dataset is loaded using `pd.read_csv()` from the provided CSV file. It is then split into features (independent variables) and the target (dependent variable). The data is further preprocessed, including handling missing values and scaling features.

## Model Building and Evaluation

Various regression models are trained and evaluated, including K-Nearest Neighbors, Decision Tree, Random Forest, AdaBoost, and Gradient Boosting regressors. Evaluation metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared score are calculated to assess model performance.

## Advanced Modeling and Model Optimization

Advanced modeling techniques like ensemble methods (Bagging, Random Forest, AdaBoost, Gradient Boosting) are explored to improve predictive performance. Hyperparameter tuning is performed using GridSearchCV to optimize model parameters. The model with the best hyperparameters is selected for final evaluation.

## Cross-Validation

Cross-validation is performed to assess the generalization performance of the selected model. The mean R-squared score across all folds is calculated to provide an overall indication of model performance.

For detailed explanations and code implementation, refer to the corresponding sections in the provided code files.

https://prezi.com/view/QVAFSjDT2Hvqae8wD6GE/
