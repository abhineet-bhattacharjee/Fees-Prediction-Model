from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from visualisation import df

X = df[['academic.year', 'school']]
y = df['cost']

ct = ColumnTransformer(
    transformers=[
        ('school_enc', OneHotEncoder(drop='first', sparse_output=False), ['school'])
    ],
    remainder='passthrough'
)
