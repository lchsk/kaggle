import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class Categorical:
    def __init__(self, name: str, df: pd.DataFrame):
        self.name = name
        self.unique_types = [[t] for t in df[name].unique()]
        self.enc = OneHotEncoder(handle_unknown='ignore')

    @property
    def feature_names(self):
        return self.enc.get_feature_names()

    def fit(self):
        self.enc.fit(self.unique_types)

    def apply(self, df: pd.DataFrame()):
        transformed_df = pd.DataFrame(data=self._transform(df), columns=self.feature_names, index=df.index)

        return pd.concat([df, transformed_df], axis=1)

    def _transform(self, df: pd.DataFrame):
        return self.enc.transform(df[self.name].values.reshape(-1, 1)).toarray()


def get_categorical_columns(df, exclude=set()):
    columns = []

    for col in df.columns:
        if col in exclude:
            continue

        if df[col].dtype == 'O':
            columns.append(col)

    return columns