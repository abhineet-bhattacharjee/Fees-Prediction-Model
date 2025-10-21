from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd

from visualisation import df_long as df


X = df[['academic.year', 'school']]
y = df['cost']

ct = ColumnTransformer(
        transformers=[
            ('school_enc', OneHotEncoder(drop='first', sparse_output=False), ['school'])
        ],
        remainder='passthrough'
    )

def encode():
    return ct.fit_transform(X)

def assign(x_encoded):
    return train_test_split(
        x_encoded, y, test_size=0.2, random_state=42
    )


if __name__ == '__main__':
    X_encoded = encode()
    assign(X_encoded)
