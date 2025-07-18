import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def preprocess_normalize_data(data, target_column):
    x_list = []
    y_list = []
    keys = [key for key in data[0].keys() if key != target_column and key != 'ID']
    for entry in data:
        item = [value for key, value in entry.items() if key != target_column and key != 'ID']
        x_list.append(np.array(item).reshape(1, -1))
        y_list.append(entry[target_column])

    x_train = np.vstack(x_list).astype(float)
    y_train = np.array(y_list, dtype=float).reshape(-1, 1)    
    return x_train, y_train, keys


def train_evaluate_linear_regression(x, y):
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

    sgdr = SGDRegressor(max_iter=10000)
    sgdr.fit(X_train_scaled, y_train.ravel())

    y_train_pred = sgdr.predict(X_train_scaled)
    y_test_pred = sgdr.predict(X_test_scaled)

    train_mse = np.mean((y_train.ravel() - y_train_pred) ** 2)
    train_rmse = np.sqrt(train_mse)
    train_mae = np.mean(np.abs(y_train.ravel() - y_train_pred))

    test_mse = np.mean((y_test.ravel() - y_test_pred) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(y_test.ravel() - y_test_pred))

    print("LINEAR REGRESSION TRAIN SET:")
    print("MSE:", train_mse)
    print("RMSE:", train_rmse)
    print("MAE:", train_mae)

    print("\nLINEAR REGRESSION TEST SET:")
    print("MSE:", test_mse)
    print("RMSE:", test_rmse)
    print("MAE:", test_mae)

    return sgdr, X_test, y_test, y_test_pred


def train_evaluate_polynomial_regression(x, y):
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
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    scaler = StandardScaler()
    X_train_poly_scaled = scaler.fit_transform(X_train_poly)
    X_test_poly_scaled = scaler.transform(X_test_poly)

    sgdr = SGDRegressor(max_iter=10000)
    sgdr.fit(X_train_poly_scaled, y_train.ravel())

    y_train_pred = sgdr.predict(X_train_poly_scaled)
    y_test_pred = sgdr.predict(X_test_poly_scaled)

    train_mse = np.mean((y_train.ravel() - y_train_pred) ** 2)
    train_rmse = np.sqrt(train_mse)
    train_mae = np.mean(np.abs(y_train.ravel() - y_train_pred))

    test_mse = np.mean((y_test.ravel() - y_test_pred) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(y_test.ravel() - y_test_pred))


    print("\nPOLYNOMIAL REGRESSION TRAIN SET:")
    # MSE (Mean Squared Error):
    # Average of the squared differences between actual and predicted prices on the training data. Lower is better.
    print("MSE:", train_mse)
    # RMSE (Root Mean Squared Error):
    # Square root of MSE. It’s in the same units as your target (thousands of dollars).
    print("RMSE:", train_rmse)
    # MAE (Mean Absolute Error):
    # Average of the absolute differences between actual and predicted prices.
    print("MAE:", train_mae)

    print("\nPOLYNOMIAL REGRESSION TEST SET:")
    print("MSE:", test_mse)
    print("RMSE:", test_rmse)
    print("MAE:", test_mae)
    return sgdr, X_test, y_test, y_test_pred


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

def train_evaluate_zscore_normalization(x_train, y_train):
    """
    Trains and evaluates a model after Z-score normalization.
    - Normalizes the input features.
    - Visualizes the distribution of selected features before and after normalization.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    sgdr = SGDRegressor(max_iter=10000)
    sgdr.fit(X_train, y_train.ravel())

    # Predict
    y_train_pred = sgdr.predict(X_train)
    y_test_pred = sgdr.predict(X_test)

    # Evaluate
    train_mse = np.mean((y_train.ravel() - y_train_pred) ** 2)
    train_rmse = np.sqrt(train_mse)
    train_mae = np.mean(np.abs(y_train.ravel() - y_train_pred))

    test_mse = np.mean((y_test.ravel() - y_test_pred) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(y_test.ravel() - y_test_pred))

    print("\nZSCORE NORMALIZED TRAIN SET:")
    # MSE (Mean Squared Error):
    # Average of the squared differences between actual and predicted prices on the training data. Lower is better.
    print("MSE:", train_mse)
    # RMSE (Root Mean Squared Error):
    # Square root of MSE. It’s in the same units as your target (thousands of dollars).
    print("RMSE:", train_rmse)
    # MAE (Mean Absolute Error):
    # Average of the absolute differences between actual and predicted prices.
    print("MAE:", train_mae)

    print("\nZSCORE NORMALIZED TEST SET:")
    print("MSE:", test_mse)
    print("RMSE:", test_rmse)
    print("MAE:", test_mae)
    return sgdr, X_test, y_test, y_test_pred

