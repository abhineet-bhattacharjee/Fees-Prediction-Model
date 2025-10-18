from pprint import pprint
import warnings

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import pandas as pd

from encoding import assign, X, ct

results = {}
X_train, X_test, y_train, y_test = assign(X)
X_train = pd.DataFrame(X_train, columns=['academic.year', 'school'])
X_test = pd.DataFrame(X_test, columns=['academic.year', 'school'])

warnings.filterwarnings(action='ignore', category=UserWarning, message='Singular matrix in solving dual problem.')
warnings.filterwarnings(action='ignore', category=RuntimeWarning, message='Ill-conditioned matrix')
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

def lasso_regression(degree):
    lasso_pipe = Pipeline([
        ('preprocess', ct),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('model', Lasso(max_iter=10000))
    ])
    lasso_params = {'model__alpha': [0.01, 0.1, 1, 10]}
    lasso_grid = GridSearchCV(lasso_pipe, lasso_params, cv=5,
                              scoring='neg_mean_absolute_error', n_jobs=-1)
    lasso_grid.fit(X_train, y_train)
    lasso_best = lasso_grid.best_estimator_
    lasso_pred = lasso_best.predict(X_test)
    lasso_mae = mean_absolute_error(y_test, lasso_pred)
    lasso_r2 = r2_score(y_test, lasso_pred)
    lasso_acc = evaluation(y_test, lasso_pred)
    results['Lasso Regression'] = {'MAE': lasso_mae, 'R2': lasso_r2, 'Accuracy (within 10%)': lasso_acc, 'Degree': degree}

def random_forest(degree):
    rf_pipe = Pipeline([
        ('preprocess', ct),
        ('model', RandomForestRegressor(random_state=42))
    ])
    rf_params = {'model__n_estimators': [50, 100], 'model__max_depth': [None, 10, 20]}
    rf_grid = GridSearchCV(rf_pipe, rf_params, cv=5,
                           scoring='neg_mean_absolute_error', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    rf_pred = rf_best.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_acc = evaluation(y_test, rf_pred)
    results['Random Forest'] = {'MAE': rf_mae, 'R2': rf_r2, 'Accuracy (within 10%)': rf_acc, 'Degree': degree}

def linear_regression(degree):
    lr_pipe = Pipeline([
        ('preprocess', ct),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('model', LinearRegression())
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    mae_lr = mean_absolute_error(y_test, lr_pred)
    r2_lr = r2_score(y_test, lr_pred)
    acc_lr = evaluation(y_test, lr_pred)
    results['Linear Regression'] = {'MAE': mae_lr, 'R2': r2_lr, 'Accuracy (within 10%)': acc_lr, 'Degree': degree}


if __name__ == '__main__':
    for degree in [1, 2]:
        ridge_regression(degree)
        lasso_regression(degree)
        random_forest(degree)
        linear_regression(degree)
    results_sorted = dict(sorted(results.items(), key=lambda x: x[1]['MAE']))
    pprint(results_sorted)
