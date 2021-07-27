#!/usr/bin/env python3

import os
import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn import linear_model
# from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# noinspection PyProtectedMember
from sklearn.linear_model._base import LinearModel
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer


# ------------------------ Useful methods and classes ------------------------ #

class TrainData:
    def __init__(self, x, y, df: DataFrame):
        self.df = df
        self.X = x
        self.y = y


# ---------------------------- Read external files --------------------------- #

project_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(os.path.dirname(project_dir), 'input')
commonlitreadabilityprize_input_dir = os.path.join(input_dir, 'commonlitreadabilityprize')
custom_input_dir = os.path.join(input_dir, 'custom')
output_dir = os.path.join(project_dir, 'out')


# ---------------------------- Data Preparation 1 ---------------------------- #

def data_prep_1(df: DataFrame, out_filename: str) -> TrainData:
    if os.path.isfile(os.path.join(output_dir, out_filename)):
        with open(os.path.join(output_dir, out_filename)) as csv_fp:
            df = pd.read_csv(csv_fp)
            # The X:
            X = df
    else:
        X = df['excerpt']

        # create the transform
        vectorizer = TfidfVectorizer(
            stop_words='english',
            # token_pattern=r"(?u)\b\w\w+\b"
            token_pattern=r'\b[^\d\W]+\b'
            # token_pattern=r"\b[^\d\W]+\b/g"
        )
        # tokenize and build vocab
        vectorizer.fit(X)
        # encode document
        X_vector: csr_matrix = vectorizer.transform(X)
        X_vectorized = pd.DataFrame(
            X_vector.toarray(),
            columns=vectorizer.get_feature_names()
        )

        X_words_count = []
        X_words_freq = []
        X_words_freq_count_ratio = []

        # Calculate words count and frequency for each record
        for index, row in X_vectorized.iterrows():
            row: pd.Series
            row.value_counts()
            words_count = 0
            words_freq = 0
            for word_freq, word_count in row.value_counts().iteritems():
                if word_freq == 0:
                    continue
                words_count += word_count
                words_freq += word_freq
            X_words_count.append(words_count)
            X_words_freq.append(words_freq)
            X_words_freq_count_ratio.append(words_freq / words_count)

        # Add the new variables to the main DataFrame
        df['words_count'] = X_words_count
        df['words_freq'] = X_words_freq
        df['words_freq_count_ratio'] = X_words_freq_count_ratio

        # Scaling numeric variables
        transformers = [
            [
                'scaler',
                RobustScaler(),
                [
                    'words_count',
                    'words_freq',
                    'words_freq_count_ratio',
                ]
            ],
        ]
        ct = ColumnTransformer(
            transformers,
            remainder='passthrough'
        )
        df = ct.fit_transform(df)
        df = DataFrame(
            data=df,
            columns=[
                'words_count',
                'words_freq',
                'words_freq_count_ratio',
                'id',
                'url_legal',
                'license',
                'excerpt',
                'target',
                'standard_error',
            ]
        )

        # The X:
        X = X_vectorized
        X['words_freq_count_ratio'] = df['words_freq_count_ratio']
        X['words_count'] = df[['words_count']]
        X['words_freq'] = df[['words_freq']]

        # Save new DataSet
        X.to_csv(os.path.join(output_dir, out_filename), index=False)

    # The y:
    y = df['target']

    return TrainData(X, y, df)


with open(os.path.join(commonlitreadabilityprize_input_dir, 'train.csv')) as train_csv_fp:
    train_csv_df: DataFrame = pd.read_csv(train_csv_fp)
    train_data_1 = data_prep_1(train_csv_df, 'train_4_1.csv')

with open(os.path.join(commonlitreadabilityprize_input_dir, 'test.csv')) as test_csv_fp:
    test_csv_df: DataFrame = pd.read_csv(test_csv_fp)
    test_data_1 = data_prep_1(train_csv_df, 'test_4_1.csv')


# -------------------------------- Modelling 1 ------------------------------- #

def train_model(
        model: LinearModel,
        train_data: TrainData
) -> LinearModel:
    print(f'\n---------- {type(model)} ----------')
    X_train, X_test, y_train, y_test = train_test_split(
        train_data.X,
        train_data.y,
        random_state=0,
        test_size=.10
    )
    model.fit(X_train, y_train)
    p_train = model.predict(X_train)
    p_test = model.predict(X_test)
    mse_train = mean_squared_error(y_train, p_train)
    mse_test = mean_squared_error(y_test, p_test)
    print(f'Median target {np.median(train_data.y)}')
    print(f'Train {mse_train}, test {mse_test}')
    # sns.scatterplot(x=X_train['words_freq_count_ratio'].values, y=y_train)
    # sns.scatterplot(x=X_test['words_freq_count_ratio'].values, y=y_test)
    # sns.lineplot(x=train_data.X['words_freq_count_ratio'].values, y=train_data.y)
    # plt.show()
    return model


def test_model(
        model: LinearModel,
        train_data: TrainData
):
    print(f'\n---------- {type(model)} ----------')
    X_test, y_test = (train_data.X, train_data.y)
    p_test = model.predict(X_test)
    mse_test = mean_squared_error(y_test, p_test)
    print(f'Median target {np.median(train_data.y)}')
    print(f'Final Test {mse_test}')
    # sns.scatterplot(x=X_train['words_freq_count_ratio'].values, y=y_train)
    # sns.scatterplot(x=X_test['words_freq_count_ratio'].values, y=y_test)
    # sns.lineplot(x=train_data.X['words_freq_count_ratio'].values, y=train_data.y)
    # plt.show()


# Train LinearRegression
trained_model = train_model(linear_model.LinearRegression(n_jobs=16), train_data_1)
# Test LinearRegression
test_model(trained_model, test_data_1)


# -------------------------------- Evaluation -------------------------------- #


# -------------------------------- Deployment -------------------------------- #

sys.exit(0)
