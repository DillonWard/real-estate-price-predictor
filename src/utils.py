import matplotlib.pyplot as plt


def draw_scatter_plot(x, y, predictions, xlabel, ylabel, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color='blue', label='Actual Prices', alpha=0.7)
    plt.scatter(
        x, predictions, color='red',
        label='Predicted Prices', alpha=0.7
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def populate_overall_results(index, results, overall_results):
    dictionary = {
        "MSE Before": 'train_mse',
        "RMSE Before": 'train_rmse',
        "MAE Before": 'train_mae',
        "MSE After": 'test_mse',
        "RMSE After": 'test_rmse',
        "MAE After": 'test_mae'
    }
    for key, value in dictionary.items():
        if value in results:
            overall_results[index][key] = results[value]
    return overall_results
