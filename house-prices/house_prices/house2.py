import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper


DATA = '../data/'

mapper = DataFrameMapper([
    ('MSSubClass', None),
    ('LotArea', None),
    ('LotShape', LabelBinarizer()),
])

features = [
    'MSSubClass',
    'LotArea',
    'LotShape',
]

pipeline = Pipeline([
    ('featurize', mapper),
    ('regr', LogisticRegression(fit_intercept=True)),
])

fts = [
    'MSSubClass',
    'LotArea',
    'LotShape',
]

fts1 = fts + ['SalePrice']

def load():
    df_train = pd.read_csv(DATA + 'train.csv', header = 0, index_col = 'Id')
    df_test = pd.read_csv(DATA + 'test.csv', header = 0, index_col = 'Id')

    return df_train, df_test


def main():
    df_train, df_test = load()

    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
