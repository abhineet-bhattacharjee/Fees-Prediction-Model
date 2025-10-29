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

from visualisation import rmse, mape, acc_within_pct

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
