from pprint import pprint
import warnings

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import pandas as pd

from encoding import assign, X, ct

results = {}
X_train, X_test, y_train, y_test = assign(X)
X_train = pd.DataFrame(X_train, columns=['academic.year', 'school'])
X_test = pd.DataFrame(X_test, columns=['academic.year', 'school'])

warnings.filterwarnings(action='ignore', category=UserWarning, message='Singular matrix in solving dual problem.')
warnings.filterwarnings(action='ignore', category=RuntimeWarning, message='Ill-conditioned matrix')

def evaluation(y_true, y_pred, tolerance=0.1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred) <= tolerance * y_true)

def ridge_regression(degree):
    ridge_pipe = Pipeline([
        ('preprocess', ct),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('model', Ridge())
    ])
    ridge_params = {'model__alpha': [0.01, 0.1, 1, 10]}
    ridge_grid = GridSearchCV(ridge_pipe, ridge_params, cv=5,
                              scoring='neg_mean_absolute_error', n_jobs=-1)
    ridge_grid.fit(X_train, y_train)
    ridge_best = ridge_grid.best_estimator_
    ridge_pred = ridge_best.predict(X_test)
    ridge_mae = mean_absolute_error(y_test, ridge_pred)
    ridge_r2 = r2_score(y_test, ridge_pred)
    ridge_acc = evaluation(y_test, ridge_pred)
    results['Ridge Regression'] = {'MAE': ridge_mae, 'R2': ridge_r2, 'Accuracy (within 10%)': ridge_acc, 'Degree': degree}


if __name__ == '__main__':
    for degree in [1, 2, 3]:
        ridge_regression(degree)
    results_sorted = dict(sorted(results.items(), key=lambda x: x[1]['MAE']))
    pprint(results_sorted)
