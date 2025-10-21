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