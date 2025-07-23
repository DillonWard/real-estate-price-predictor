from src.read import read_csv
from src.models import (
    preprocess_normalize_data,
    zscore_normalize,
    tune_linear_hyperparameters,
    tune_polynomial_hyperparameters,
    tune_zscore_hyperparameters,
    tune_rf_hyperparameters,
    train_evaluate_linear_regression,
    train_evaluate_polynomial_regression,
    train_evaluate_zscore_normalization,
    train_evaluate_random_forest,
)
import concurrent.futures
from src.vars import columns_to_ignore


def train_linear(x_train, y_train, model_name):
    params = tune_linear_hyperparameters(
        x_train,
        y_train,
        save=True,
        suffix=(model_name.replace("linear_regression", "")),
    )
    train_evaluate_linear_regression(
        x_train, y_train, params=params, model_name=model_name
    )


def train_polynomial(x_train, y_train, model_name):
    params = tune_polynomial_hyperparameters(
        x_train,
        y_train,
        save=True,
        suffix=model_name.replace("polynomial_regression", ""),
    )
    train_evaluate_polynomial_regression(
        x_train, y_train, params=params, model_name=model_name
    )


def train_zscore(x_train, y_train, model_name):
    x_norm, mean, sigma = zscore_normalize(x_train)
    params = tune_zscore_hyperparameters(
        x_norm,
        y_train,
        save=True,
        suffix=model_name.replace("zscore_regression", "")
    )
    train_evaluate_zscore_normalization(
        x_norm, y_train, params=params, model_name=model_name
    )


def train_rf(x_train, y_train, model_name):
    params = tune_rf_hyperparameters(
        x_train,
        y_train,
        save=True,
        suffix=model_name.replace("random_forest_regression", ""),
    )
    train_evaluate_random_forest(
        x_train, y_train, params=params, model_name=model_name, save=True
    )


def main():
    data = read_csv("data/train.csv")
    feature_configs = [
        {"name": "full", "columns_to_ignore": ["ID"]},
        {"name": "less", "columns_to_ignore": columns_to_ignore},
    ]
    model_types = [
        ("linear_regression", train_linear),
        ("polynomial_regression", train_polynomial),
        ("zscore_regression", train_zscore),
        ("random_forest_regression", train_rf),
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for config in feature_configs:
            x_train, y_train = preprocess_normalize_data(
                data=data,
                target_column="medv",
                columns_to_ignore=config["columns_to_ignore"],
            )
            suffix = "" if config["name"] == "full" else "_less_features"
            for model_type, func in model_types:
                model_name = f"{model_type}{suffix}"
                futures.append(executor.submit(
                    func,
                    x_train,
                    y_train,
                    model_name
                ))
        for future in concurrent.futures.as_completed(futures):
            future.result()


main()
