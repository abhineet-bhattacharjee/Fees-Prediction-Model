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
