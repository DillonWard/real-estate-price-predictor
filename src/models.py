import json
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import joblib
from src.read import load_model, load_params

def preprocess_normalize_data(data, target_column):
    x_list = []
    y_list = []
    for entry in data:
        item = [value for key, value in entry.items() if key != target_column and key != 'ID']
        x_list.append(np.array(item).reshape(1, -1))
        y_list.append(entry[target_column])

    x_train = np.vstack(x_list).astype(float)
    y_train = np.array(y_list, dtype=float).reshape(-1, 1)    
    return x_train, y_train


def make_predictions_and_evaluate(model, x_train, y_train, x_test, y_test):
    """
    Makes predictions using the provided model and evaluates the performance.
    - Calculates MSE, RMSE, and MAE for both training and test sets.
    - Prints the evaluation metrics.
    """
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_mse = np.mean((y_train.ravel() - y_train_pred) ** 2)
    train_rmse = np.sqrt(train_mse)
    train_mae = np.mean(np.abs(y_train.ravel() - y_train_pred))

    test_mse = np.mean((y_test.ravel() - y_test_pred) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(y_test.ravel() - y_test_pred))

    print("\nTRAIN SET:")
    print("MSE:", train_mse)
    print("RMSE:", train_rmse)
    print("MAE:", train_mae)

    print("\nTEST SET:")
    print("MSE:", test_mse)
    print("RMSE:", test_rmse)
    print("MAE:", test_mae)

    results = {
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }

    return y_test_pred, results

def train_evaluate_linear_regression(x, y, params={'max_iter': 10000}):
    """
    Trains a linear regression model using SGDRegressor on the provided data.
    - Splits the data into training and test sets.
    - Scales features using StandardScaler.
    - Fits the model on the training set.
    - Makes predictions on both train and test sets.
    - Calculates and prints MSE, RMSE, and MAE for both sets.
    - Returns the trained model and test set predictions.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sgdr = load_model('data/linear_regression_pipeline.joblib')
    if sgdr is None:
        sgdr = SGDRegressor(**params)
        sgdr.fit(X_train_scaled, y_train.ravel())

    y_test_pred, results = make_predictions_and_evaluate(sgdr, X_train_scaled, y_train, X_test_scaled, y_test)

    return sgdr, X_test, y_test, y_test_pred, results

def train_evaluate_polynomial_regression(x, y, params={'max_iter': 10000}):
    """
    Trains a polynomial regression model (linear regression on polynomial features).
    - Splits the data into training and test sets.
    - Expands features to polynomial terms (degree=2).
    - Scales the expanded features.
    - Fits the model on the training set.
    - Makes predictions on both train and test sets.
    - Calculates and prints MSE, RMSE, and MAE for both sets.
    - Returns the trained model and test set predictions.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    pipeline = load_model('data/polynomial_regression_pipeline.joblib')
    if pipeline is None:
        degree = params.pop('poly__degree', 2)
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        scaler = StandardScaler()
        model = SGDRegressor(**params)
        pipeline = Pipeline([
            ('poly', poly),
            ('scaler', scaler),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_train.ravel())
    y_test_pred, results = make_predictions_and_evaluate(
        pipeline, X_train, y_train, X_test, y_test
    )

    return pipeline, X_test, y_test, y_test_pred, results

def zscore_normalize(x):
    """
    Z-score normalization of the input data.
    - Computes the mean and standard deviation for each feature.
    - Normalizes each feature by subtracting the mean and dividing by the standard deviation.
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    X_norm = (x - mean) / std
    return X_norm, mean, std

def train_evaluate_zscore_normalization(x, y, params={'max_iter': 10000}):
    """
    Trains and evaluates a model after Z-score normalization.
    - Normalizes the input features.
    - Visualizes the distribution of selected features before and after normalization.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    sgdr = load_model('data/zscale_normalized_pipeline.joblib')
    if sgdr is None:
        sgdr = SGDRegressor(**params)
        sgdr.fit(X_train, y_train.ravel())

    y_test_pred, results = make_predictions_and_evaluate(
        sgdr, X_train, y_train, X_test, y_test
    )

    return sgdr, X_test, y_test, y_test_pred, results

def perform_hyperparameter_tuning(x_train, y_train, pipeline_type=None, save=False):
    if pipeline_type == 'polynomial':
        stored_params = load_params('data/polynomial_best_params.json')
        if stored_params is not None:
            return stored_params
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('regressor', SGDRegressor())
        ])
        param_grid = {
            'poly__degree': [2, 3, 4],
            'regressor__alpha': [0.0001, 0.001, 0.01, 0.1],
            'regressor__max_iter': [10000, 50000],
            'regressor__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'regressor__eta0': [0.01, 0.1, 1.0]
        }
    elif pipeline_type == 'linear':
        stored_params = load_params('data/linear_best_params.json')
        if stored_params is not None:
            return stored_params
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SGDRegressor())
        ])
        param_grid = {
            'regressor__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'regressor__max_iter': [5000, 10000, 20000],
            'regressor__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'regressor__eta0': [0.001, 0.01, 0.1, 1.0],
            'regressor__tol': [1e-4, 1e-3, 1e-2]
        }
    else:
        stored_params = load_params('data/zscale_best_params.json')
        if stored_params is not None:
            return stored_params
        pipeline = SGDRegressor()
        param_grid = {
            'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'max_iter': [5000, 10000, 20000],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'eta0': [0.001, 0.01, 0.1, 1.0],
            'tol': [1e-4, 1e-3, 1e-2]
        }

    grid = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid.fit(x_train, y_train.ravel())

    best_params = {k.replace('regressor__', ''): v for k, v in grid.best_params_.items()}

    if save:
        if pipeline_type == 'polynomial':
            joblib.dump(grid, 'data/polynomial_regression_pipeline.joblib')
            with open('data/polynomial_best_params.json', 'w') as f:
                json.dump(best_params, f)
        elif pipeline_type == 'linear':
            joblib.dump(grid, 'data/linear_regression_pipeline.joblib')
            with open('data/linear_best_params.json', 'w') as f:
                json.dump(best_params, f)
        else:
            joblib.dump(grid, 'data/zscale_normalized_pipeline.joblib')
            with open('data/zscale_best_params.json', 'w') as f:
                json.dump(best_params, f)

    return best_params


def train_evaluate_random_forest(x, y, params=None, save=False):
    """
    Trains and evaluates a RandomForestRegressor.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    pipeline_path = 'data/random_forest_pipeline.joblib'
    pipeline = load_model(pipeline_path)

    if pipeline is None:
        if params is None:
            params = {}
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(**params, random_state=42))
        ])
        pipeline.fit(X_train, y_train.ravel())
        if save:
            joblib.dump(pipeline, pipeline_path)

    y_test_pred, results = make_predictions_and_evaluate(
        pipeline, X_train, y_train, X_test, y_test
    )

    return pipeline, X_test, y_test, y_test_pred, results

def perform_rf_hyperparameter_tuning(x_train, y_train, save=False):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid.fit(x_train_scaled, y_train.ravel())
    best_params = {k.replace('regressor__', ''): v for k, v in grid.best_params_.items()}
    if save:
        with open('data/random_forest_best_params.json', 'w') as f:
            json.dump(best_params, f)

    return best_params