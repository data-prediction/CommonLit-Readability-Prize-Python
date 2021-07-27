#!/usr/bin/env python3

import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
# noinspection PyProtectedMember
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# noinspection PyProtectedMember
from sklearn.neural_network._multilayer_perceptron import BaseMultilayerPerceptron
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer


# ------------------------ Useful methods and classes ------------------------ #

class TrainData:
    def __init__(self, x, y):
        self.X = x
        self.y = y


def evaluate_model(
        model: LinearModel or BaseMultilayerPerceptron,
        train_data: TrainData
):
    print(f'\n---------- {type(model)} ----------')

    X_train, X_test, y_train, y_test = train_test_split(
        train_data.X,
        train_data.y,
        random_state=0,
        test_size=.20
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


# ---------------------------- Read external files --------------------------- #

project_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(os.path.dirname(project_dir), 'input')
commonlitreadabilityprize_input_dir = os.path.join(input_dir, 'commonlitreadabilityprize')
custom_input_dir = os.path.join(input_dir, 'custom')
output_dir = os.path.join(project_dir, 'out')

with open(os.path.join(commonlitreadabilityprize_input_dir, 'train.csv')) as train_csv_fp:
    train_csv_df: DataFrame = pd.read_csv(train_csv_fp)

with open(os.path.join(commonlitreadabilityprize_input_dir, 'train.csv')) as test_csv_fp:
    test_csv_df: DataFrame = pd.read_csv(test_csv_fp)

with open(os.path.join(custom_input_dir, 'unigram_freq.csv')) as unigram_freq_fp:
    unigram_freq_df: DataFrame = pd.read_csv(unigram_freq_fp, index_col='word')
    unigram_freq_dict: dict = unigram_freq_df.to_dict().get('count')

if os.path.isfile(os.path.join(output_dir, 'trained_1.csv')):
    with open(os.path.join(output_dir, 'trained_1.csv')) as trained_csv_fp_1:
        trained_csv_df_1: DataFrame or None = pd.read_csv(trained_csv_fp_1)
else:
    trained_csv_df_1 = None

if os.path.isfile(os.path.join(output_dir, 'trained_2.csv')):
    with open(os.path.join(output_dir, 'trained_2.csv')) as trained_csv_fp_2:
        trained_csv_df_2: DataFrame or None = pd.read_csv(trained_csv_fp_2)
else:
    trained_csv_df_2 = None

if os.path.isfile(os.path.join(output_dir, 'trained_3.csv')):
    with open(os.path.join(output_dir, 'trained_3.csv')) as trained_csv_fp:
        trained_csv_df_3: DataFrame or None = pd.read_csv(trained_csv_fp)
else:
    trained_csv_df_3 = None


# ---------------------------- Data Preparation 1 ---------------------------- #

def data_prep_1() -> TrainData:
    global train_csv_df

    if trained_csv_df_1 is None:
        X = train_csv_df['excerpt']

        # create the transform
        vectorizer = CountVectorizer(
            stop_words='english',
            # token_pattern=r"(?u)\b\w\w+\b"
            token_pattern=r'\b[^\d\W]+\b'
            # token_pattern=r"\b[^\d\W]+\b/g"
        )
        # tokenize and build vocab
        vectorizer.fit(X)
        # encode document
        X_vector: csr_matrix = vectorizer.transform(X)
        X_vectorized = pd.DataFrame(X_vector.toarray(), columns=vectorizer.get_feature_names())

        X_words_count = []
        X_words_freq = []
        X_words_freq_count_ratio = []

        # Calculate words count and frequency for each record
        for index, row in X_vectorized.iterrows():
            row: pd.Series
            words_count = 0
            words_freq = 0
            for word, word_count in row.iteritems():
                if word_count == 0:
                    continue
                word_freq = unigram_freq_dict.get(word)
                if word_freq is None:
                    continue
                words_count += word_count
                words_freq += word_freq
            X_words_count.append(words_count)
            X_words_freq.append(words_freq)
            X_words_freq_count_ratio.append(words_freq / words_count)

        # Add the new variables to the main DataFrame
        train_csv_df['words_count'] = X_words_count
        train_csv_df['words_freq'] = X_words_freq
        train_csv_df['words_freq_count_ratio'] = X_words_freq_count_ratio

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
        train_csv_df = ct.fit_transform(train_csv_df)
        train_csv_df = DataFrame(
            data=train_csv_df,
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

        # Save new DataSet
        train_csv_df.to_csv(os.path.join(output_dir, 'trained_1.csv'), index=False)
    else:
        train_csv_df = trained_csv_df_1

    # The y:
    y = train_csv_df['target'].values

    # The X:
    X = DataFrame(
        data=train_csv_df[[
            'words_count',
            'words_freq',
            'words_freq_count_ratio'
        ]],
        columns=[
            'words_count',
            'words_freq',
            'words_freq_count_ratio'
        ]
    )

    return TrainData(X, y)


# -------------------------------- Modelling 1 ------------------------------- #

train_data_1 = data_prep_1()

evaluate_model(LinearRegression(), train_data_1)

evaluate_model(
    MLPRegressor(hidden_layer_sizes=[2, 4, 2], max_iter=200, tol=-1, verbose=False),
    train_data_1
)


# ---------------------------- Data Preparation 2 ---------------------------- #

def data_prep_2() -> TrainData:
    global train_csv_df

    if trained_csv_df_2 is None:
        X = train_csv_df['excerpt']

        # create the transform
        vectorizer = TfidfVectorizer(
            stop_words='english',
            token_pattern=r'\b[^\d\W]+\b'
        )
        # tokenize and build vocab
        vectorizer.fit(X)
        # encode document
        X_vector: csr_matrix = vectorizer.transform(X)
        X_vectorized = pd.DataFrame(X_vector.toarray(), columns=vectorizer.get_feature_names())

        X_words_internal_freq = []

        # Calculate words count and frequency for each record
        for index, row in X_vectorized.iterrows():
            row: pd.Series
            words_internal_freq = 0
            for word, word_internal_freq in row.iteritems():
                if word_internal_freq == 0:
                    continue
                words_internal_freq += word_internal_freq
            X_words_internal_freq.append(words_internal_freq)

        # Add the new variables to the main DataFrame
        train_csv_df['words_internal_freq'] = X_words_internal_freq

        # Scaling numeric variables
        transformers = [
            [
                'scaler',
                RobustScaler(),
                [
                    'words_count',
                    'words_freq',
                    'words_freq_count_ratio',
                    'words_internal_freq'
                ]
            ],
        ]
        ct = ColumnTransformer(
            transformers,
            remainder='passthrough'
        )
        train_csv_df = ct.fit_transform(train_csv_df)
        train_csv_df = DataFrame(
            data=train_csv_df,
            columns=[
                'words_count',
                'words_freq',
                'words_freq_count_ratio',
                'words_internal_freq',
                'id',
                'url_legal',
                'license',
                'excerpt',
                'target',
                'standard_error',
            ]
        )

        # Save new DataSet
        train_csv_df.to_csv(os.path.join(output_dir, 'trained_2.csv'), index=False)
    else:
        train_csv_df = trained_csv_df_2

    # The y:
    y = train_csv_df['target'].values

    # The X:
    X = DataFrame(
        data=train_csv_df[[
            'words_count',
            'words_internal_freq'
        ]],
        columns=[
            'words_count',
            'words_internal_freq'
        ]
    )

    return TrainData(X, y)


# -------------------------------- Modelling 2 ------------------------------- #

train_data_2 = data_prep_2()
evaluate_model(
    MLPRegressor(hidden_layer_sizes=[2, 4, 2], max_iter=200, tol=-1, verbose=False),
    train_data_2
)


# ---------------------------- Data Preparation 3 ---------------------------- #

def data_prep_3() -> TrainData:
    global train_csv_df

    if trained_csv_df_3 is None:
        X = train_csv_df['excerpt']
        # create the transform
        vectorizer = HashingVectorizer(
            stop_words='english',
            token_pattern=r'\b[^\d\W]+\b',
            n_features=2**11
        )
        # tokenize and build vocab
        vectorizer.fit(X)
        # encode document
        X_vector: csr_matrix = vectorizer.transform(X)
        X_vectorized = pd.DataFrame(X_vector.toarray())
        X_vectorized.to_csv(os.path.join(output_dir, 'trained_3.csv'), index=False)
    else:
        X_vectorized = trained_csv_df_3

    # The y:
    y = train_csv_df['target'].values

    # The X:
    X = DataFrame(
        data=X_vectorized
    )
    X['words_count'] = train_csv_df[['words_count']]
    X['words_internal_freq'] = train_csv_df[['words_internal_freq']]

    return TrainData(X, y)


# -------------------------------- Modelling 3 ------------------------------- #

train_data_3 = data_prep_3()
evaluate_model(
    MLPRegressor(
        hidden_layer_sizes=[2],
        max_iter=60,
        tol=-1
    ),
    train_data_3
)


# -------------------------------- Evaluation -------------------------------- #


# -------------------------------- Deployment -------------------------------- #

exit(0)
