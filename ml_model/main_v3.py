#!/usr/bin/env python3

import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network._multilayer_perceptron import BaseMultilayerPerceptron
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer


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

if os.path.isfile(os.path.join(output_dir, 'train.csv')):
    with open(os.path.join(output_dir, 'train.csv')) as trained_csv_fp:
        trained_csv_df: DataFrame or None = pd.read_csv(trained_csv_fp)
else:
    trained_csv_df = None


# ---------------------------- Data Preparation 1 ---------------------------- #

if trained_csv_df is None:
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
    train_csv_df = DataFrame(
        data=ct.fit_transform(train_csv_df),
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
    train_csv_df.to_csv(os.path.join(output_dir, 'train.csv'))
else:
    train_csv_df = trained_csv_df


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


# -------------------------------- Modelling 1 ------------------------------- #

def evaluate_model(model: LinearModel or BaseMultilayerPerceptron):
    global X
    global y

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=.15)

    model.fit(X_train, y_train)

    p_train = model.predict(X_train)
    p_test = model.predict(X_test)

    mse_train = mean_squared_error(y_train, p_train)
    mse_test = mean_squared_error(y_test, p_test)

    print(f'Median target {np.median(y)}')
    print(f'Train {mse_train}, test {mse_test}')

    sns.scatterplot(x=X_train['words_freq_count_ratio'].values, y=y_train)
    sns.scatterplot(x=X_test['words_freq_count_ratio'].values, y=y_test)
    sns.lineplot(x=X['words_freq_count_ratio'].values, y=y)
    plt.show()


evaluate_model(LinearRegression())
evaluate_model(
    MLPRegressor(hidden_layer_sizes=[2, 4, 2], max_iter=10000, tol=-1, verbose=False)
)


# ---------------------------- Data Preparation 2 ---------------------------- #


# -------------------------------- Evaluation -------------------------------- #


# -------------------------------- Deployment -------------------------------- #

exit(0)
