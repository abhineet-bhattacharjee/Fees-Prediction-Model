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

from model_selection import rmse, mape, acc_within_pct, build_poly_model, eval_all
from visualisation import df


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

def train_and_save_models():
    school_cols = [c for c in df.columns if c != 'academic.year']
    ensure_dir(MODELS_DIR)
    report = {'cv_results': {}, 'train_fit': {}}

    for school in school_cols:
        spec = BEST_MODELS[school]
        degree = spec['degree']
        params = spec['params']
        X = df[['academic.year']]
        y = df[school]
        pipe = build_poly_model(LinearRegression(**params), degree)
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_mae = -cross_val_score(pipe, X, y, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
        cv_r2 = cross_val_score(pipe, X, y, scoring='r2', cv=kf, n_jobs=-1)
        pipe.fit(X, y)
        y_pred = pipe.predict(X)

        train_metrics = eval_all(y, y_pred)
        train_metrics['Degree'] = degree
        report['cv_results'][school] = {
            'CV_MAE_mean': float(cv_mae.mean()),
            'CV_MAE_std': float(cv_mae.std()),
            'CV_R2_mean': float(cv_r2.mean()),
            'CV_R2_std': float(cv_r2.std()),
            'Degree': degree
        }

        report['train_fit'][school] = train_metrics
        model_path = os.path.join(MODELS_DIR, f'final_model_{school.replace(" ", "_")}.joblib')
        joblib.dump(pipe, model_path)

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f'Models saved to: {MODELS_DIR}')
    print(f'Report saved to: {REPORT_PATH}')

def predict(school, year):
    model_path = os.path.join(MODELS_DIR, f'final_model_{school.replace(" ", "_")}.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found for school: {school}')

    model = joblib.load(model_path)
    X_new = pd.DataFrame({'academic.year': [year]})
    y_pred = model.predict(X_new)
    return float(y_pred[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--school', type=str, default=None)
    parser.add_argument('--year', type=int, default=None)
    args = parser.parse_args()

    if args.train:
        train_and_save_models()
    elif args.predict:
        if args.school is None or args.year is None:
            raise ValueError('Provide --school "School Name" and --year YYYY')
        y_pred = predict(args.school, args.year)
        print(f'Predicted tuition for {args.school} in {args.year}: {y_pred:.2f}')
    else:
        print('Use --train to train/save models or --predict --school "Name" --year YYYY to predict.')


if __name__ == '__main__':
    main()
