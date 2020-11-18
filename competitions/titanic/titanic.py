from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import preprocessing

from my.submissions.submission import save_submission
from my.cleaner.categorical import Categorical
from my.dirs import join_paths, get_data_dir

COMPETITION = 'titanic'

@dataclass()
class Var:
    cat_sex = None

    def fit(self):
        self.cat_sex.fit()


var = Var()


def get_feature_columns(df):
    return df[[
        'Pclass',
        'Age',
        'SibSp',
        'Parch',
        'Fare',

        'x0_female',
        'x0_male',
    ]]


def process_data(df):
    df.Age = df.Age.fillna(df.Age.mean())
    df.Fare = df.Fare.fillna(df.Fare.mean())
    df.Fare = preprocessing.scale(df.Fare)

    le = LabelEncoder()
    le.fit(df.Sex.unique())
    df['SexValue'] = le.transform(df.Sex)

    df = var.cat_sex.apply(df)

    return df


def train(df):
    y = df.iloc[:, 1]
    X = get_feature_columns(df)

    clf = LogisticRegressionCV(cv=10, random_state=0).fit(X, y)

    print('score', clf.score(X, y))

    return clf


def run():
    data_dir = get_data_dir(COMPETITION, __file__)
    df_train = pd.read_csv(join_paths(data_dir, 'train.csv'))
    df_test = pd.read_csv(join_paths(data_dir, 'test.csv'))

    var.cat_sex = Categorical('Sex', df_train)
    var.fit()

    df_train = process_data(df_train)
    df_test = process_data(df_test)

    clf = train(df_train)

    predicted = clf.predict(get_feature_columns(df_test))

    save_submission(columns=['PassengerId', 'Survived'], data=[df_test.PassengerId, predicted])

if __name__ == '__main__':
    run()

