import argparse
import json
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORT_PATH = os.path.join(BASE_DIR, 'model_report.json')
RANDOM_STATE = 42
CV_FOLDS = 5

BEST_MODELS = {
    'Business (MBA)': {'model': 'LinearRegression', 'degree': 3, 'params': {}},
    'Design': {'model': 'LinearRegression', 'degree': 3, 'params': {}},
    'Divinity': {'model': 'LinearRegression', 'degree': 3, 'params': {}},
    'Education': {'model': 'LinearRegression', 'degree': 2, 'params': {}},
    'GSAS': {'model': 'LinearRegression', 'degree': 3, 'params': {}},
    'Government': {'model': 'LinearRegression', 'degree': 3, 'params': {}},
    'Law': {'model': 'LinearRegression', 'degree': 3, 'params': {}},
    'Medical/Dental': {'model': 'LinearRegression', 'degree': 3, 'params': {}},
    'Public Health (1-Year MPH)': {'model': 'LinearRegression', 'degree': 3, 'params': {}},
}

def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def acc_within_pct(y_true, y_pred, pct=0.10):
    return float((np.abs(np.asarray(y_true)-np.asarray(y_pred)) <= pct*np.clip(np.abs(np.asarray(y_true)), 1e-8, None)).mean())

def build_pipeline(degree, params):
    base = LinearRegression(**params)
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scale', StandardScaler()),
        ('model', base)
    ])

def train_and_save_models():
    ensure_dir(MODELS_DIR)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f'Cannot find dataset at {DATA_PATH}')
    df = pd.read_csv(DATA_PATH)
    school_cols = [c for c in df.columns if c != 'academic.year']
    report = {'cv_results': {}, 'train_fit': {}}
    for school in school_cols:
        if school not in BEST_MODELS:
            raise KeyError(f'Missing BEST_MODELS entry for school: {school}')
        spec = BEST_MODELS[school]
        degree = spec['degree']
        params = spec['params']
        X = df[['academic.year']]
        y = df[school]
        pipe = build_pipeline(degree, params)
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_mae = -cross_val_score(pipe, X, y, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
        cv_r2 = cross_val_score(pipe, X, y, scoring='r2', cv=kf, n_jobs=-1)
        pipe.fit(X, y)
        y_hat = pipe.predict(X)
        train_metrics = {
            'MAE': float(mean_absolute_error(y, y_hat)),
            'RMSE': rmse(y, y_hat),
            'R2': float(r2_score(y, y_hat)),
            'MAPE_%': mape(y, y_hat),
            'Acc_within_10%': acc_within_pct(y, y_hat, 0.10),
            'Degree': degree
        }
        report['cv_results'][school] = {
            'CV_MAE_mean': float(cv_mae.mean()),
            'CV_MAE_std': float(cv_mae.std()),
            'CV_R2_mean': float(cv_r2.mean()),
            'CV_R2_std': float(cv_r2.std()),
            'Degree': degree
        }
        report['train_fit'][school] = train_metrics
        fname = f'final_model_{safe_name(school)}.joblib'
        model_path = os.path.join(MODELS_DIR, fname)
        print(f'Saving: {model_path}')
        joblib.dump(pipe, model_path)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'Report saved to: {REPORT_PATH}')

def predict(school: str, year: int) -> float:
    fname = f'final_model_{safe_name(school)}.joblib'
    model_path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found for school: {school} at {model_path}')
    model = joblib.load(model_path)
    X_new = pd.DataFrame({'academic.year': [year]})
    y_pred = model.predict(X_new)
    return float(y_pred[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--school', type=str)
    parser.add_argument('--year', type=int)
    args = parser.parse_args()
    if args.train:
        train_and_save_models()
    elif args.predict:
        if not args.school or args.year is None:
            raise ValueError('Use --predict --school "School Name" --year YYYY')
        yhat = predict(args.school, args.year)
        print(f'Predicted tuition for {args.school} in {args.year}: {yhat:.2f}')
    else:
        print('Use --train to train/save models or --predict --school "Name" --year YYYY to predict.')

if __name__ == '__main__':
    main()
