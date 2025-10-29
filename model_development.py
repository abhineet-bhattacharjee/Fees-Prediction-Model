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
