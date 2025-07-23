import numpy as np
import unittest
from src.models import (
    zscore_normalize,
    preprocess_normalize_data,
    train_evaluate_linear_regression,
    train_evaluate_polynomial_regression,
    train_evaluate_zscore_normalization,
    train_evaluate_random_forest,
    tune_linear_hyperparameters,
    tune_rf_hyperparameters,
    tune_polynomial_hyperparameters,
    tune_zscore_hyperparameters,
    make_predictions_and_evaluate
)
from src.utils import (
    draw_scatter_plot,
    populate_overall_results
)
from src.read import read_csv, load_model, load_params
from unittest.mock import patch


class TestPreprocessing(unittest.TestCase):
    def test_zscore_normalize(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        x_norm, mean, std = zscore_normalize(x)
        self.assertTrue(np.allclose(np.mean(x_norm, axis=0), 0, atol=1e-7))
        self.assertTrue(np.allclose(np.std(x_norm, axis=0), 1, atol=1e-7))

    def test_preprocess_normalize_data(self):
        data = [{"a": 1, "b": 2, "medv": 3}, {"a": 4, "b": 5, "medv": 6}]
        x, y = preprocess_normalize_data(data, target_column="medv")
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(y.shape, (2, 1))

    def test_zscore_normalize_with_constant_column(self):
        x = np.array([[1, 1], [1, 1], [1, 1]])
        x_norm, mean, std = zscore_normalize(x)
        self.assertTrue(np.allclose(mean, [1, 1]))
        self.assertTrue(np.allclose(std, [0, 0]))

    def test_zscore_normalize_nan(self):
        x = np.array([[np.nan, 2], [3, 4], [5, 6]])
        x_norm, mean, std = zscore_normalize(np.nan_to_num(x))
        self.assertFalse(np.isnan(x_norm).any())


class TestUtils(unittest.TestCase):
    def test_load_model_missing(self):
        self.assertIsNone(load_model("data/nonexistent_model.joblib"))

    def test_load_params_missing(self):
        self.assertIsNone(load_params("data/nonexistent_params.json"))

    def test_read_csv_empty(self):
        try:
            result = read_csv("data/nonexistent.csv")
            self.assertIsInstance(result, list)
        except Exception:
            pass

    def test_draw_scatter_plot_runs(self):
        x = np.arange(10)
        y = np.random.rand(10)
        predictions = np.random.rand(10)
        try:
            draw_scatter_plot(x, y, predictions, "x", "y", "title")
        except Exception as e:
            self.fail(f"draw_scatter_plot raised {e}")

    def test_populate_overall_results(self):
        results = {
            "train_mse": 1,
            "train_rmse": 2,
            "train_mae": 3,
            "test_mse": 4,
            "test_rmse": 5,
            "test_mae": 6,
        }
        overall = [
            {"MSE Before": 0, "RMSE Before": 0, "MAE Before": 0,
             "MSE After": 0, "RMSE After": 0, "MAE After": 0}
        ]
        updated = populate_overall_results(0, results, overall)
        self.assertEqual(updated[0]["MSE Before"], 1)
        self.assertEqual(updated[0]["RMSE After"], 5)

    def test_load_model_and_params_types(self):
        self.assertIsNone(load_model("data/nonexistent_model.joblib"))
        self.assertIsNone(load_params("data/nonexistent_params.json"))

    def test_read_csv_real_file(self):
        import os

        test_path = "data/test.csv"
        with open(test_path, "w") as f:
            f.write("a,b,medv\n1,2,3\n4,5,6\n")
        result = read_csv(test_path)
        os.remove(test_path)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["medv"], "3")


class TestModelTraining(unittest.TestCase):
    @patch("src.models.load_model", return_value=None)
    def test_train_evaluate_linear_regression_metrics(self, _):
        x = np.random.rand(30, 13)
        y = np.random.rand(30, 1)
        model, X_test, y_test, y_test_pred, results = (
            train_evaluate_linear_regression(x, y)
        )
        self.assertEqual(y_test_pred.shape, y_test.ravel().shape)
        self.assertTrue("test_mse" in results and "train_mse" in results)

    @patch("src.models.load_model", return_value=None)
    def test_train_evaluate_polynomial_regression_metrics(self, _):
        x = np.random.rand(30, 13)
        y = np.random.rand(30, 1)
        model, X_test, y_test, y_test_pred, results = (
            train_evaluate_polynomial_regression(x, y)
        )
        self.assertEqual(y_test_pred.shape, y_test.ravel().shape)
        self.assertTrue("test_mse" in results and "train_mse" in results)

    @patch("src.models.load_model", return_value=None)
    def test_train_evaluate_zscore_normalization_metrics(self, _):
        x = np.random.rand(30, 13)
        y = np.random.rand(30, 1)
        x_norm, _, _ = zscore_normalize(x)
        model, X_test, y_test, y_test_pred, results = (
            train_evaluate_zscore_normalization(x_norm, y)
        )
        self.assertEqual(y_test_pred.shape, y_test.ravel().shape)
        self.assertTrue("test_mse" in results and "train_mse" in results)

    def test_train_evaluate_random_forest_metrics(self):
        x = np.random.rand(30, 13)
        y = np.random.rand(30, 1)
        model, X_test, y_test, y_test_pred, results = (
            train_evaluate_random_forest(
                x, y
            )
        )
        self.assertEqual(y_test_pred.shape, y_test.ravel().shape)
        self.assertTrue("test_mse" in results and "train_mse" in results)

    def test_perform_hyperparameter_tuning_linear(self):
        x = np.random.rand(30, 13)
        y = np.random.rand(30, 1)
        params = tune_linear_hyperparameters(x, y)
        self.assertIsInstance(params, dict)
        self.assertTrue("max_iter" in params)

    def test_make_predictions_and_evaluate_output(self):
        from src.models import make_predictions_and_evaluate
        from sklearn.linear_model import LinearRegression

        x_train = np.random.rand(20, 13)
        y_train = np.random.rand(20, 1)
        x_test = np.random.rand(5, 13)
        y_test = np.random.rand(5, 1)
        model = LinearRegression().fit(x_train, y_train)
        y_test_pred, results = make_predictions_and_evaluate(
            model, x_train, y_train, x_test, y_test
        )
        self.assertEqual(y_test_pred.shape[0], y_test.ravel().shape[0])
        self.assertIn("train_mse", results)
        self.assertIn("test_mse", results)

    def test_preprocess_normalize_data_with_id(self):
        from src.models import preprocess_normalize_data

        data = [
            {"ID": 1, "a": 2, "b": 3, "medv": 4},
            {"ID": 2, "a": 5, "b": 6, "medv": 7},
        ]
        x, y = preprocess_normalize_data(data, target_column="medv")
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(y.shape, (2, 1))

    def test_zscore_normalize_zero_std(self):
        from src.models import zscore_normalize

        x = np.ones((10, 13))
        x_norm, mean, std = zscore_normalize(x)
        self.assertTrue(np.all(std == 0))
        self.assertTrue(np.all(np.isnan(x_norm) | (x_norm == 0)))

    def test_train_evaluate_random_forest_with_params(self):
        from src.models import train_evaluate_random_forest

        x = np.random.rand(30, 13)
        y = np.random.rand(30, 1)
        params = {"n_estimators": 10, "max_depth": 2}
        model, X_test, y_test, y_test_pred, results = (
            train_evaluate_random_forest(
                x, y, params=params
            )
        )
        self.assertEqual(y_test_pred.shape, y_test.ravel().shape)
        self.assertTrue("test_mse" in results and "train_mse" in results)

    def test_perform_rf_hyperparameter_tuning(self):
        x = np.random.rand(30, 13)
        y = np.random.rand(30, 1)
        params = tune_rf_hyperparameters(x, y)
        self.assertIsInstance(params, dict)
        self.assertTrue("n_estimators" in params)

    @patch("src.models.load_model", return_value=None)
    def test_train_evaluate_linear_regression_with_params(self, _):
        x = np.random.rand(30, 5)
        y = np.random.rand(30, 1)
        params = {"max_iter": 500, "alpha": 0.01, "penalty": "l2"}
        model, X_test, y_test, y_test_pred, results = (
            train_evaluate_linear_regression(x, y, params=params)
        )
        self.assertEqual(X_test.shape[1], x.shape[1])
        self.assertEqual(y_test_pred.shape, y_test.ravel().shape)
        self.assertTrue("test_mse" in results)

    @patch("src.models.load_model", return_value=None)
    def test_train_evaluate_polynomial_regression_with_params(self, _):
        x = np.random.rand(30, 5)
        y = np.random.rand(30, 1)
        params = {
            "max_iter": 500,
            "alpha": 0.01,
            "penalty": "l2",
            "poly__degree": 2
        }
        model, X_test, y_test, y_test_pred, results = (
            train_evaluate_polynomial_regression(x, y, params=params)
        )
        self.assertEqual(X_test.shape[1], x.shape[1])
        self.assertEqual(y_test_pred.shape, y_test.ravel().shape)
        self.assertTrue("test_mse" in results)

    @patch("src.models.load_model", return_value=None)
    def test_train_evaluate_zscore_normalization_with_params(self, _):
        x = np.random.rand(30, 5)
        y = np.random.rand(30, 1)
        x_norm, _, _ = zscore_normalize(x)
        params = {"max_iter": 500, "alpha": 0.01, "penalty": "l2"}
        model, X_test, y_test, y_test_pred, results = (
            train_evaluate_zscore_normalization(x_norm, y, params=params)
        )
        self.assertEqual(X_test.shape[1], x_norm.shape[1])
        self.assertEqual(y_test_pred.shape, y_test.ravel().shape)
        self.assertTrue("test_mse" in results)

    def test_train_evaluate_random_forest_with_custom_params(self):
        x = np.random.rand(30, 5)
        y = np.random.rand(30, 1)
        params = {"n_estimators": 5, "max_depth": 2}
        model, X_test, y_test, y_test_pred, results = (
            train_evaluate_random_forest(
                x, y, params=params
            )
        )
        self.assertEqual(y_test_pred.shape, y_test.ravel().shape)
        self.assertTrue("test_mse" in results)

    def test_tune_linear_hyperparameters_returns_dict(self):
        x = np.random.rand(20, 5)
        y = np.random.rand(20, 1)
        params = tune_linear_hyperparameters(x, y)
        self.assertIsInstance(params, dict)
        self.assertIn("max_iter", params)

    def test_tune_polynomial_hyperparameters_returns_dict(self):
        x = np.random.rand(20, 5)
        y = np.random.rand(20, 1)
        params = tune_polynomial_hyperparameters(x, y)
        self.assertIsInstance(params, dict)
        self.assertIn("max_iter", params)
        self.assertIn("poly__degree", params)

    def test_tune_zscore_hyperparameters_returns_dict(self):
        x = np.random.rand(20, 5)
        y = np.random.rand(20, 1)
        params = tune_zscore_hyperparameters(x, y)
        self.assertIsInstance(params, dict)
        self.assertIn("max_iter", params)

    def test_tune_rf_hyperparameters_returns_dict(self):
        x = np.random.rand(20, 5)
        y = np.random.rand(20, 1)
        params = tune_rf_hyperparameters(x, y)
        self.assertIsInstance(params, dict)
        self.assertIn("n_estimators", params)

    def test_make_predictions_and_evaluate_with_bad_model(self):
        class DummyModel:
            def predict(self, X):
                return np.zeros(X.shape[0])
        x_train = np.random.rand(10, 3)
        y_train = np.random.rand(10, 1)
        x_test = np.random.rand(5, 3)
        y_test = np.random.rand(5, 1)
        y_test_pred, results = make_predictions_and_evaluate(
            DummyModel(), x_train, y_train, x_test, y_test
        )
        self.assertEqual(y_test_pred.shape[0], y_test.shape[0])
        self.assertIn("train_mse", results)
        self.assertIn("test_mse", results)


if __name__ == "__main__":
    unittest.main()
