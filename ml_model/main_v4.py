#!/usr/bin/env python3

import os
import sys

import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer


# ------------------------ Useful methods and classes ------------------------ #

class TrainData:
    def __init__(self, df: DataFrame, X: DataFrame, y: DataFrame = None):
        self.df = df
        self.X = X
        self.y = y


# ---------------------- External files and directories ---------------------- #

project_dir = os.path.realpath(os.getcwd())
input_dir = os.path.join(project_dir, 'input')
commonlitreadabilityprize_input_dir = os.path.join(input_dir, 'commonlitreadabilityprize')
output_dir = os.path.join(project_dir, 'ml_model', 'out')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


# ---------------------------- Data Preparation 1 ---------------------------- #

def data_prep_1(df_orig: DataFrame, out_filename: str) -> TrainData:
    df = df_orig.copy()

    # The y:
    if 'target' in df.columns:
        y = df['target']
    else:
        y = None

    out_filepath = os.path.join(output_dir, out_filename)
    if os.path.isfile(out_filepath):
        with open(out_filepath) as csv_fp:
            df = pd.read_csv(csv_fp)
            # The X:
            X = df
    else:
        df_columns = list(df.columns)
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
        # df.columns
        df = ct.fit_transform(df)
        df = DataFrame(
            data=df,
            columns=[
                        'words_count',
                        'words_freq',
                        'words_freq_count_ratio'
                    ] + df_columns
        )

        # The X:
        X = X_vectorized
        X['words_freq_count_ratio'] = df['words_freq_count_ratio']
        X['words_count'] = df[['words_count']]
        X['words_freq'] = df[['words_freq']]

        # Save new DataSet
        X.to_csv(out_filepath, index=False)

    return TrainData(df_orig, X, y,)


with open(os.path.join(commonlitreadabilityprize_input_dir, 'train.csv')) as train_csv_fp:
    train_csv_df: DataFrame = pd.read_csv(train_csv_fp)
    # Prepare data for Training
    train_data_1 = data_prep_1(train_csv_df, 'train_4_1.csv')

with open(os.path.join(commonlitreadabilityprize_input_dir, 'test.csv')) as test_csv_fp:
    # Prepare data for Testing
    test_csv_df: DataFrame = pd.read_csv(test_csv_fp)
    # Prepare data for Testing
    test_data_1 = data_prep_1(test_csv_df, 'test_4_1.csv')

# Standardize test and train columns
col_list = list(set().union(test_data_1.X.columns, train_data_1.X.columns))
test_data_1.X = test_data_1.X.reindex(columns=col_list, fill_value=0)
train_data_1.X = train_data_1.X.reindex(columns=col_list, fill_value=0)


# -------------------------------- Modelling 1 ------------------------------- #

def train_model(
        model: linear_model.LinearRegression,
        train_data: TrainData
) -> linear_model.LinearRegression:
    print(f'\n---------- Training {type(model)} ----------')
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
    return model


# Train and Test LinearRegression
trained_model = train_model(linear_model.LinearRegression(n_jobs=16), train_data_1)


# -------------------------------- Evaluation -------------------------------- #

def evaluate_model(
        model: linear_model.LinearRegression,
        test_data: TrainData
) -> DataFrame:
    print(f'\n---------- Evaluating {type(model)} ----------')
    p_test = model.predict(test_data.X)
    p_df = DataFrame()
    p_df['id'] = test_data.df[['id']]
    p_df['target'] = p_test
    print(p_df)
    return p_df


# Evaluate LinearRegression model
result_df = evaluate_model(trained_model, test_data_1)


# -------------------------------- Deployment -------------------------------- #

result_df.to_csv(os.path.join(output_dir, 'result_4.csv'), index=False)

sys.exit(0)
