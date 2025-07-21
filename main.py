from src.read import read_csv
from src.models import preprocess_normalize_data, train_evaluate_linear_regression, train_evaluate_polynomial_regression

def main():
    '''
        crim
        per capita crime rate by town.

        zn
        proportion of residential land zoned for lots over 25,000 sq.ft.

        indus
        proportion of non-retail business acres per town.

        chas
        Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

        nox
        nitrogen oxides concentration (parts per 10 million).

        rm
        average number of rooms per dwelling.

        age
        proportion of owner-occupied units built prior to 1940.

        dis
        weighted mean of distances to five Boston employment centres.

        rad
        index of accessibility to radial highways.

        tax
        full-value property-tax rate per $10,000.

        ptratio
        pupil-teacher ratio by town.

        black
        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

        lstat
        lower status of the population (percent).

        medv
        median value of owner-occupied homes in $1000s.
    '''
    data = read_csv('data/train.csv')
    x_train, y_train = preprocess_normalize_data(data=data, target_column='medv')
    train_evaluate_linear_regression(x_train, y_train)
    train_evaluate_polynomial_regression(x_train, y_train)

main()