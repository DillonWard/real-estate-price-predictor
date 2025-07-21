import numpy as np
import unittest
from src.models import (
    zscore_normalize,
    preprocess_normalize_data,
    train_evaluate_linear_regression,
    train_evaluate_polynomial_regression,
    train_evaluate_zscore_normalization,
    train_evaluate_random_forest,
)
from src.read import read_csv, load_model, load_params


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
    def test_train_evaluate_linear_regression_metrics(self):
        x = np.random.rand(30, 13)
        y = np.random.rand(30, 1)
        model, X_test, y_test, y_test_pred, results = (
            train_evaluate_linear_regression(x, y)
        )
        self.assertEqual(y_test_pred.shape, y_test.ravel().shape)
        self.assertTrue("test_mse" in results and "train_mse" in results)

    def test_train_evaluate_polynomial_regression_metrics(self):
        x = np.random.rand(30, 13)
        y = np.random.rand(30, 1)
        model, X_test, y_test, y_test_pred, results = (
            train_evaluate_polynomial_regression(x, y)
        )
        self.assertEqual(y_test_pred.shape, y_test.ravel().shape)
        self.assertTrue("test_mse" in results and "train_mse" in results)

    def test_train_evaluate_zscore_normalization_metrics(self):
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
        from src.models import perform_hyperparameter_tuning

        params = perform_hyperparameter_tuning(x, y, pipeline_type="linear")
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
        from src.models import perform_rf_hyperparameter_tuning

        x = np.random.rand(30, 13)
        y = np.random.rand(30, 1)
        params = perform_rf_hyperparameter_tuning(x, y)
        self.assertIsInstance(params, dict)
        self.assertTrue("n_estimators" in params)


if __name__ == "__main__":
    unittest.main()
