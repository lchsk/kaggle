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

df_train = pd.read_csv(DATA + 'train.csv', header = 0, index_col = 'Id')[fts1]
df_test = pd.read_csv(DATA + 'test.csv', header = 0, index_col = 'Id')[fts]

df = pd.concat([df_train, df_test], keys=["train", "test"])

df_train = df.ix['train']
df_test  = df.ix['test']

# X = df_train[df_train.columns.drop('SalePrice')]
X = df_train
y = df_train['SalePrice']

model = pipeline.fit(X = X, y = y)

pred = model.predict(df_test)

score = cross_val_score(model, X, y, 'mean_squared_error')

print(score)

pd.DataFrame(dict(
    Id=df_test.index,
    SalePrice=pred,
)).to_csv('out.csv', index=False)
