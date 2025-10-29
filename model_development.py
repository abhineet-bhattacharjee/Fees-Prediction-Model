import argparse
import json
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
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
    'Business (MBA)': {'model': 'LinearRegression', 'degree': 2, 'params': {}},
    'Design': {'model': 'LinearRegression', 'degree': 2, 'params': {}},
    'Divinity': {'model': 'Lasso', 'degree': 3, 'params': {'alpha': 0.0005}},
    'Education': {'model': 'LinearRegression', 'degree': 2, 'params': {}},
    'GSAS': {'model': 'LinearRegression', 'degree': 2, 'params': {}},
    'Government': {'model': 'Lasso', 'degree': 3, 'params': {'alpha': 1}},
    'Law': {'model': 'LinearRegression', 'degree': 2, 'params': {}},
    'Medical/Dental': {'model': 'Lasso', 'degree': 3, 'params': {'alpha': 0.001}},
    'Public Health (1-Year MPH)': {'model': 'Ridge', 'degree': 3, 'params': {'alpha': 1}},
}


def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def train_and_save_models():
    school_cols = [c for c in df.columns if c not in ['academic.year', 'inflation_rate', 'endowment_billions']]
    ensure_dir(MODELS_DIR)
    report = {'cv_results': {}, 'train_fit': {}}

    for school in school_cols:
        spec = BEST_MODELS[school]
        model_type = spec['model']
        degree = spec['degree']
        params = spec['params']

        if model_type == 'LinearRegression':
            base_model = LinearRegression(**params)
        elif model_type == 'Lasso':
            base_model = Lasso(max_iter=50000, **params)
        elif model_type == 'Ridge':
            base_model = Ridge(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        X = df[['academic.year', 'inflation_rate', 'endowment_billions']]
        y = df[school]

        pipe = build_poly_model(base_model, degree)

        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_mae = -cross_val_score(pipe, X, y, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
        cv_r2 = cross_val_score(pipe, X, y, scoring='r2', cv=kf, n_jobs=-1)

        pipe.fit(X, y)
        y_pred = pipe.predict(X)

        train_metrics = eval_all(y, y_pred)
        train_metrics['Degree'] = degree
        train_metrics['Model'] = model_type

        report['cv_results'][school] = {
            'Model': model_type,
            'CV_MAE_mean': float(cv_mae.mean()),
            'CV_MAE_std': float(cv_mae.std()),
            'CV_R2_mean': float(cv_r2.mean()),
            'CV_R2_std': float(cv_r2.std()),
            'Degree': degree
        }

        report['train_fit'][school] = train_metrics

        model_path = os.path.join(MODELS_DIR, f'final_model_{safe_name(school.replace(" ", "_"))}.joblib')
        joblib.dump(pipe, model_path)

        print(f"Trained {school}: {model_type} (degree {degree}) | MAE={train_metrics['MAE']:.2f}, R2={train_metrics['R2']:.4f}")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f'\nAll models saved to: {MODELS_DIR}')
    print(f'Report saved to: {REPORT_PATH}')


def predict(school, year, inflation=None, endowment=None):
    model_path = os.path.join(MODELS_DIR, f'final_model_{school.replace(" ", "_")}.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found for school: {school}')

    if inflation is None:
        inflation = df['inflation_rate'].iloc[-1]
        print(f"  Using default inflation: {inflation:.2f}%")

    if endowment is None:
        recent_years = df['academic.year'].iloc[-5:].values
        recent_endow = df['endowment_billions'].iloc[-5:].values
        slope = (recent_endow[-1] - recent_endow[0]) / (recent_years[-1] - recent_years[0])
        endowment = recent_endow[-1] + slope * (year - recent_years[-1])
        print(f"  Estimated endowment: ${endowment:.2f}B")

    model = joblib.load(model_path)
    X_new = pd.DataFrame({
        'academic.year': [year],
        'inflation_rate': [inflation],
        'endowment_billions': [endowment]
    })

    y_pred = model.predict(X_new)
    return float(y_pred[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train and save all models')
    parser.add_argument('--predict', action='store_true', help='Predict tuition for a school')
    parser.add_argument('--school', type=str, default=None, help='School name')
    parser.add_argument('--year', type=int, default=None, help='Academic year')
    parser.add_argument('--inflation', type=float, default=None, help='Inflation rate (%). Optional.')
    parser.add_argument('--endowment', type=float, default=None, help='Endowment in billions. Optional.')
    args = parser.parse_args()

    if args.train:
        train_and_save_models()
    elif args.predict:
        if args.school is None or args.year is None:
            raise ValueError('Provide --school "School Name" and --year YYYY')
        y_pred = predict(args.school, args.year, args.inflation, args.endowment)
        print(f'\n✓ Predicted tuition for {args.school} in {args.year}: ${y_pred:,.2f}')
    else:
        print('Usage:')
        print('  --train                          Train and save models')
        print('  --predict --school "Name" --year YYYY [--inflation X.X] [--endowment Y.Y]')




if __name__ == '__main__':
    main()
