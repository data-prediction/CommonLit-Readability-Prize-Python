#!/usr/bin/env python3

import os
import sys

import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from tidytext import unnest_tokens, bind_tf_idf
# noinspection PyProtectedMember
from siuba import _, count, arrange
from nltk.tokenize import word_tokenize


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
custom_input_dir = os.path.join(input_dir, 'custom')
output_dir = os.path.join(project_dir, 'ml_model', 'out')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


# ---------------------------- Data Preparation 1 ---------------------------- #

def data_prep_1(df_orig: DataFrame, out_filename_main: str, out_filename_X: str) -> TrainData:
    # The y:
    if 'target' in df_orig.columns:
        y = df_orig['target']
    else:
        y = None

    out_filepath_main = os.path.join(output_dir, out_filename_main)
    out_filepath_X = os.path.join(output_dir, out_filename_X)

    if os.path.isfile(out_filepath_main) and os.path.isfile(out_filepath_X):
        with open(out_filepath_main) as csv_fp:
            # Main CSV at this step:
            df_orig = pd.read_csv(csv_fp)
        with open(out_filepath_X) as csv_fp:
            # The X:
            X = pd.read_csv(csv_fp)
    else:
        df_columns = list(df_orig.columns)
        X = df_orig['excerpt']
        # create the transform
        vectorizer = TfidfVectorizer(
            stop_words='english',
            # token_pattern=r"(?u)\b\w\w+\b"
            token_pattern=r'\b[^\d\W]+\b'
            # token_pattern=r"\b[^\d\W]+\b/g"
        )
        # tokenize and build vocab
        vectorizer.fit(X)
        vector_columns = vectorizer.get_feature_names()
        # encode document
        X_vector: csr_matrix = vectorizer.transform(X)

        # The X:
        X = pd.DataFrame(
            X_vector.toarray(),
            columns=vector_columns
        )

        X_words_count = []
        X_words_freq = []
        X_words_freq_count_ratio = []

        # Calculate words count and frequency for each record
        for index, row in X.iterrows():
            row: pd.Series
            row.value_counts()
            words_freq = 0
            words_count = 0
            for word_freq, word_count in row.value_counts().iteritems():
                if word_freq == 0:
                    continue
                words_freq += word_freq
                words_count += word_count
            X_words_count.append(words_count)
            X_words_freq.append(words_freq)
            X_words_freq_count_ratio.append(words_freq / words_count)

        # Add the new variables to the main DataFrame
        df_orig['words_count'] = X_words_count
        df_orig['words_freq'] = X_words_freq
        df_orig['words_freq_count_ratio'] = X_words_freq_count_ratio

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
        df_orig = ct.fit_transform(df_orig)
        df_orig = DataFrame(
            data=df_orig,
            columns=[
                        'words_count',
                        'words_freq',
                        'words_freq_count_ratio',
                    ] + df_columns
        )

        # Save new DataSet
        # noinspection PyTypeChecker
        X.to_csv(out_filepath_X, index=False)
        # noinspection PyTypeChecker
        df_orig.to_csv(out_filepath_main, index=False)

    return TrainData(df_orig, X, y)


with open(os.path.join(commonlitreadabilityprize_input_dir, 'train.csv')) as train_csv_fp:
    train_csv_df: DataFrame = pd.read_csv(train_csv_fp)
    # Prepare data for Training
    train_data_1 = data_prep_1(train_csv_df, 'train_4_1_main.csv', 'train_4_1_train.csv')

with open(os.path.join(commonlitreadabilityprize_input_dir, 'test.csv')) as test_csv_fp:
    # Prepare data for Testing
    test_csv_df: DataFrame = pd.read_csv(test_csv_fp)
    # Prepare data for Testing
    test_data_1 = data_prep_1(test_csv_df, 'test_4_1_main.csv', 'test_4_1_test.csv')

# Standardize test and train columns

col_list_X = list(set().union(test_data_1.X.columns, train_data_1.X.columns))
col_list_main = list(set().union(test_data_1.df.columns, train_data_1.df.columns))

train_data_1.X = train_data_1.X.reindex(columns=col_list_X, fill_value=0)
test_data_1.X = test_data_1.X.reindex(columns=col_list_X, fill_value=0)

train_data_1.df = train_data_1.df.reindex(columns=col_list_main, fill_value=0)
test_data_1.df = test_data_1.df.reindex(columns=col_list_main, fill_value=0)


def tokenizer(text: str) -> list:
    english_stopwords = []  # stopwords.words('english')
    return [
        w.lower() for w in word_tokenize(text) \
        if len(w) > 3 and w not in english_stopwords and w.isalpha()
    ]


def cleaner(text: str) -> str:
    tokens = tokenizer(text)
    return ' '.join(tokens)


def data_prep_2(orig_df: DataFrame, out_filename: str) -> TrainData:
    # The y:
    if 'target' in orig_df.columns:
        y = orig_df['target']
    else:
        y = None

    out_filepath = os.path.join(output_dir, out_filename)
    if os.path.isfile(out_filepath):
        with open(out_filepath) as csv_fp:
            # Main CSV at this step:
            X_final = pd.read_csv(csv_fp)
            return TrainData(orig_df, X_final, y)

    vector_size = 30

    orig_df['excerpt_tokenized'] = orig_df['excerpt'].apply(tokenizer)
    orig_df['excerpt_cleaned'] = orig_df['excerpt'].apply(cleaner)
    X_unnested: DataFrame = (
            orig_df
            >> unnest_tokens(_.word, _.excerpt)
            >> count(_.id, _.word)
            >> bind_tf_idf(_.word, _.id, _.n)
            >> arrange(-_.tf_idf)
    )

    word_2_vec_model = Word2Vec(
        orig_df['excerpt_tokenized'],
        min_count=1,
        vector_size=vector_size,
        window=5,
        # hs=1
    )

    embedding_df = DataFrame(
        data=word_2_vec_model.wv.index_to_key,
        columns=['word']
    )
    for i in range(0, vector_size):
        embedding_df[f'V{i + 1}'] = word_2_vec_model.wv.vectors[:, i]

    X_unnested_embedding = pd.merge(
        embedding_df,
        X_unnested,
        on='word'
    )

    group_dict = dict()
    for i in range(0, vector_size):
        group_dict[f'V{i+1}'] = ['mean']
    group_dict['tf_idf'] = ['mean']

    X = X_unnested_embedding.groupby('id', as_index=False).agg(group_dict)
    X.columns = X.columns.droplevel(1)
    X_final = pd.merge(
        X,
        orig_df,
        on='id'
    )

    # noinspection PyTypeChecker
    X_final.to_csv(out_filepath, index=False)

    return TrainData(orig_df, X_final, y)


train_data_2 = data_prep_2(train_data_1.df, 'train_4_2.csv')
test_data_2 = data_prep_2(test_data_1.df, 'test_4_2.csv')


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
trained_model_1 = train_model(linear_model.LinearRegression(n_jobs=16), train_data_1)
trained_model_2 = train_model(linear_model.LinearRegression(n_jobs=16), train_data_2)


# -------------------------------- Evaluation -------------------------------- #

def evaluate_model(
        model: linear_model.LinearRegression,
        test_data: TrainData
) -> DataFrame:
    print(f'\n---------- Evaluating {type(model)} ----------')
    p_test = model.predict(test_data.X)
    p_df = DataFrame(
        data=test_data.df[['id']].values,
        columns=['id']
    )
    p_df['target'] = p_test
    print(p_df)
    return p_df


# Evaluate LinearRegression model
result_df_1 = evaluate_model(trained_model_1, test_data_1)
result_df_2 = evaluate_model(trained_model_2, test_data_2)

# -------------------------------- Deployment -------------------------------- #

# noinspection PyTypeChecker
# result_df.to_csv(os.path.join(project_dir, 'submission.csv'), index=False)
