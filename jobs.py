from src.read import read_csv
from src.models import (
    preprocess_normalize_data,
    zscore_normalize,
    perform_hyperparameter_tuning,
    perform_rf_hyperparameter_tuning,
    train_evaluate_random_forest,
)
import concurrent.futures


def train_linear(x_train, y_train):
    perform_hyperparameter_tuning(
        x_train, y_train, pipeline_type='linear', save=True
        )


def train_polynomial(x_train, y_train):
    perform_hyperparameter_tuning(
        x_train, y_train, pipeline_type='polynomial', save=True
        )


def train_zscore(x_train, y_train):
    x_norm, mean, sigma = zscore_normalize(x_train)
    perform_hyperparameter_tuning(
        x_norm, y_train, pipeline_type='zscore', save=True
        )


def train_rf(x_train, y_train):
    params = perform_rf_hyperparameter_tuning(
        x_train, y_train, save=True
    )
    train_evaluate_random_forest(
        x_train, y_train, params=params, save=True
    )


def main():
    data = read_csv("data/train.csv")
    x_train, y_train = preprocess_normalize_data(
        data=data, target_column="medv"
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(train_linear, x_train, y_train),
            executor.submit(train_polynomial, x_train, y_train),
            executor.submit(train_zscore, x_train, y_train),
            executor.submit(train_rf, x_train, y_train),
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
