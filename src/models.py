import json
import numpy as np
import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from src.read import load_model, load_params


def preprocess_normalize_data(data, target_column, columns_to_ignore=["ID"]):
    x_list, y_list = [], []
    for entry in data:
        item = [
            value
            for key, value in entry.items()
            if key != target_column and key not in columns_to_ignore
        ]
        x_list.append(np.array(item).reshape(1, -1))
        y_list.append(entry[target_column])
    x_train = np.vstack(x_list).astype(float)
    y_train = np.array(y_list, dtype=float).reshape(-1, 1)
    return x_train, y_train


def zscore_normalize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / std, mean, std


def make_predictions_and_evaluate(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    def metrics(y_true, y_pred):
        mse = np.mean((y_true.ravel() - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true.ravel() - y_pred))
        return mse, rmse, mae

    train_mse, train_rmse, train_mae = metrics(y_train, y_train_pred)
    test_mse, test_rmse, test_mae = metrics(y_test, y_test_pred)
    print(
        f"\nTRAIN SET:\nMSE: {train_mse}\nRMSE: {train_rmse}\nMAE: {train_mae}"
    )
    print(f"\nTEST SET:\nMSE: {test_mse}\nRMSE: {test_rmse}\nMAE: {test_mae}")
    results = {
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
    }
    return y_test_pred, results


def tune_linear_hyperparameters(x_train, y_train, save=False, suffix=""):
    params_path = f"data/linear_best_params{suffix}.json"
    stored_params = load_params(params_path)
    if stored_params is not None:
        return stored_params
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", SGDRegressor())
    ])
    param_grid = {
        "regressor__penalty": ["l2", "l1", "elasticnet"],
        "regressor__alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
        "regressor__max_iter": [5000, 10000, 20000],
        "regressor__learning_rate": [
            "constant", "optimal", "invscaling", "adaptive"
        ],
        "regressor__eta0": [0.001, 0.01, 0.1, 1.0],
        "regressor__tol": [1e-4, 1e-3, 1e-2],
    }
    grid = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1
    )
    grid.fit(x_train, y_train.ravel())
    best_params = {
        k.replace("regressor__", ""): v for k, v in grid.best_params_.items()
    }
    if save:
        joblib.dump(grid, f"data/linear_regression{suffix}.joblib")
        with open(params_path, "w") as f:
            json.dump(best_params, f)
    return best_params


def train_evaluate_linear_regression(
    x, y, params=None, model_name="linear_regression"
):
    if params is None:
        params = {"max_iter": 10000}
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    sgdr = load_model(f"data/{model_name}.joblib")
    if sgdr is None:
        sgdr = SGDRegressor(**params)
        sgdr.fit(X_train_scaled, y_train.ravel())
        joblib.dump(sgdr, f"data/{model_name}.joblib")
    y_test_pred, results = make_predictions_and_evaluate(
        sgdr, X_train_scaled, y_train, X_test_scaled, y_test
    )
    return sgdr, X_test, y_test, y_test_pred, results


def tune_polynomial_hyperparameters(x_train, y_train, save=False, suffix=""):
    params_path = f"data/polynomial_best_params{suffix}.json"
    stored_params = load_params(params_path)
    if stored_params is not None:
        return stored_params
    pipeline = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regressor", SGDRegressor()),
        ]
    )
    param_grid = {
        "poly__degree": [2, 3, 4],
        "regressor__penalty": ["l2", "l1", "elasticnet"],
        "regressor__alpha": [0.0001, 0.001, 0.01, 0.1],
        "regressor__max_iter": [10000, 50000],
        "regressor__learning_rate": [
            "constant", "optimal", "invscaling", "adaptive"
        ],
        "regressor__eta0": [0.01, 0.1, 1.0],
    }
    grid = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1
    )
    grid.fit(x_train, y_train.ravel())
    best_params = {
        k.replace("regressor__", ""): v for k, v in grid.best_params_.items()
    }
    if save:
        joblib.dump(
            grid,
            f"data/polynomial_regression{suffix}.joblib"
        )
        with open(params_path, "w") as f:
            json.dump(best_params, f)
    return best_params


def train_evaluate_polynomial_regression(
    x, y, params=None, model_name="polynomial_regression"
):
    if params is None:
        params = {"max_iter": 10000}
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    pipeline = load_model(f"data/{model_name}.joblib")
    if pipeline is None:
        degree = params.pop("poly__degree", 2)
        pipeline = Pipeline(
            [
                (
                    "poly",
                    PolynomialFeatures(
                        degree=degree,
                        include_bias=False
                    )
                ),
                ("scaler", StandardScaler()),
                ("regressor", SGDRegressor(**params)),
            ]
        )
        pipeline.fit(X_train, y_train.ravel())
        joblib.dump(pipeline, f"data/{model_name}.joblib")
    y_test_pred, results = make_predictions_and_evaluate(
        pipeline, X_train, y_train, X_test, y_test
    )
    return pipeline, X_test, y_test, y_test_pred, results


def tune_zscore_hyperparameters(x_train, y_train, save=False, suffix=""):
    params_path = f"data/zscale_best_params{suffix}.json"
    stored_params = load_params(params_path)
    if stored_params is not None:
        return stored_params
    pipeline = SGDRegressor()
    param_grid = {
        "penalty": ["l2", "l1", "elasticnet"],
        "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
        "max_iter": [5000, 10000, 20000],
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
        "eta0": [0.001, 0.01, 0.1, 1.0],
        "tol": [1e-4, 1e-3, 1e-2],
    }
    grid = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1
    )
    grid.fit(x_train, y_train.ravel())
    best_params = grid.best_params_
    if save:
        joblib.dump(grid, f"data/zscale_normalized{suffix}.joblib")
        with open(params_path, "w") as f:
            json.dump(best_params, f)
    return best_params


def train_evaluate_zscore_normalization(
    x, y, params=None, model_name="zscale_normalized"
):
    if params is None:
        params = {"max_iter": 10000}
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    sgdr = load_model(f"data/{model_name}.joblib")
    if sgdr is None:
        sgdr = SGDRegressor(**params)
        sgdr.fit(X_train, y_train.ravel())
        joblib.dump(sgdr, f"data/{model_name}.joblib")
    y_test_pred, results = make_predictions_and_evaluate(
        sgdr, X_train, y_train, X_test, y_test
    )
    return sgdr, X_test, y_test, y_test_pred, results


def tune_rf_hyperparameters(x_train, y_train, save=False, suffix=""):
    params_path = f"data/random_forest_best_params{suffix}.json"
    stored_params = load_params(params_path)
    if stored_params is not None:
        return stored_params
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(x_train_scaled, y_train.ravel())
    best_params = grid.best_params_
    if save:
        with open(params_path, "w") as f:
            json.dump(best_params, f)
    return best_params


def train_evaluate_random_forest(
    x, y, params=None, save=False, model_name="random_forest"
):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    pipeline_path = f"data/{model_name}.joblib"
    pipeline = load_model(pipeline_path)
    if pipeline is None:
        if params is None:
            params = {}
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    RandomForestRegressor(
                        **params,
                        random_state=42
                    )
                ),
            ]
        )
        pipeline.fit(X_train, y_train.ravel())
        if save:
            joblib.dump(pipeline, pipeline_path)
    y_test_pred, results = make_predictions_and_evaluate(
        pipeline, X_train, y_train, X_test, y_test
    )
    return pipeline, X_test, y_test, y_test_pred, results
