from pprint import pprint
import warnings
import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore', category=UserWarning, message='Singular matrix in solving dual problem')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Ill-conditioned matrix')
warnings.filterwarnings('ignore', category=ConvergenceWarning)

DATA_PATH = 'dataset.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEGREES = [1, 2, 3]

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))

def mape(y_true, y_pred, eps=1e-8):
    return float((np.abs(np.asarray(y_true)-np.asarray(y_pred)) / np.clip(np.abs(np.asarray(y_true)), eps, None)).mean() * 100.0)

def acc_within_pct(y_true, y_pred, pct=0.10):
    return float((np.abs(np.asarray(y_true)-np.asarray(y_pred)) <= pct*np.clip(np.abs(np.asarray(y_true)), 1e-8, None)).mean())

def eval_all(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE_%': mape(y_true, y_pred),
        'Acc_within_10%': acc_within_pct(y_true, y_pred, 0.10)
    }

def build_poly_model(base_estimator, degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scale', StandardScaler()),
        ('model', base_estimator)
    ])

def fit_grid(model, param_grid, X_train, y_train, scoring='neg_mean_absolute_error', cv=5):
    gs = GridSearchCV(model, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_

def run_linear(name, degree, X_train, X_test, y_train, y_test):
    model = build_poly_model(LinearRegression(), degree)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    met = eval_all(y_test, pred)
    met.update({'Model': name, 'Degree': degree, 'BestParams': {}})
    return met

def run_ridge(degree, X_train, X_test, y_train, y_test):
    model = build_poly_model(Ridge(), degree)
    grid = {'model__alpha': [0.01, 0.1, 1, 10, 100]}
    best, params = fit_grid(model, grid, X_train, y_train)
    pred = best.predict(X_test)
    met = eval_all(y_test, pred)
    met.update({'Model': 'Ridge', 'Degree': degree, 'BestParams': params})
    return met


def run_lasso(degree, X_train, X_test, y_train, y_test):
    model = build_poly_model(Lasso(max_iter=50000), degree)
    grid = {'model__alpha': [0.0005, 0.001, 0.01, 0.1, 1]}
    best, params = fit_grid(model, grid, X_train, y_train)
    pred = best.predict(X_test)
    met = eval_all(y_test, pred)
    met.update({'Model': 'Lasso', 'Degree': degree, 'BestParams': params})
    return met

def run_rf(X_train, X_test, y_train, y_test):
    model = Pipeline([('model', RandomForestRegressor(random_state=RANDOM_STATE))])
    grid = {'model__n_estimators': [200, 500], 'model__max_depth': [None, 10, 20], 'model__min_samples_leaf': [1, 2]}
    best, params = fit_grid(model, grid, X_train, y_train)
    pred = best.predict(X_test)
    met = eval_all(y_test, pred)
    met.update({'Model': 'RandomForest', 'Degree': 'N/A', 'BestParams': params})
    return met


if __name__ == '__main__':
    wide_df = pd.read_csv(DATA_PATH)
    school_cols = [c for c in wide_df.columns if c != 'academic.year']
    all_results = {}
    summary = []
    for school in school_cols:
        X = wide_df[['academic.year']].copy()
        y = wide_df[school].copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
        school_metrics = []
        for d in DEGREES:
            school_metrics.append(run_linear('LinearRegression', d, X_train, X_test, y_train, y_test))
        for d in DEGREES:
            school_metrics.append(run_ridge(d, X_train, X_test, y_train, y_test))
        for d in DEGREES:
            school_metrics.append(run_lasso(d, X_train, X_test, y_train, y_test))
        school_metrics.append(run_rf(X_train, X_test, y_train, y_test))
        all_results[school] = school_metrics
        best = min(school_metrics, key=lambda m: m['MAE'])
        summary.append({
            'School': school,
            'BestModel': best['Model'],
            'Degree': best['Degree'],
            'MAE': best['MAE'],
            'RMSE': best['RMSE'],
            'R2': best['R2'],
            'MAPE_%': best['MAPE_%'],
            'Acc_within_10%': best['Acc_within_10%'],
            'BestParams': best['BestParams']
        })
    print('\n=== Best model per school (by MAE) ===')
    for row in summary:
        print(f"{row['School']}: {row['BestModel']} (degree {row['Degree']}) | MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}, R2={row['R2']:.4f}, MAPE={row['MAPE_%']:.2f}%, Acc<=10%={row['Acc_within_10%']:.3f}, Params={row['BestParams']}")
    print('\n=== Full results ===')
    pprint(all_results)