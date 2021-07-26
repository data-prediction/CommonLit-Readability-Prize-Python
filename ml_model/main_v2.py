#!/usr/bin/env python3

import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer

# ---------------------------- Read external files --------------------------- #

project_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(os.path.dirname(project_dir), 'input')
commonlitreadabilityprize_input_dir = os.path.join(input_dir, 'commonlitreadabilityprize')
custom_input_dir = os.path.join(input_dir, 'custom')

with open(os.path.join(commonlitreadabilityprize_input_dir, 'train.csv')) as train_csv_fp:
    train_csv_df = pd.read_csv(train_csv_fp)

with open(os.path.join(commonlitreadabilityprize_input_dir, 'train.csv')) as test_csv_fp:
    test_csv_df = pd.read_csv(test_csv_fp)


# ----------------------------- Text processing ------------------------------ #

# Vectorize text by occurrences
bigram_vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    min_df=1
)
X_train_bigram = bigram_vectorizer.fit_transform(train_csv_df.excerpt)

# Scale vectors
X_train_bigram_df = pd.DataFrame.sparse.from_spmatrix(
    X_train_bigram,
    columns=bigram_vectorizer.get_feature_names()
)

# Transform vectors
transformers = [
    ['scaler', RobustScaler(), bigram_vectorizer.get_feature_names()],
]
ct = ColumnTransformer(transformers, remainder='passthrough')
# X_train_transformed = ct.fit_transform(X_train_bigram_df)

training_results = {
    'X_train_bigram': {
        'features_count': bigram_vectorizer.get_feature_names(),
        'result': X_train_bigram_df
    }
}

# ------------------------------- Regression --------------------------------- #

y = train_csv_df.target


def learn(trained_results):
    X_train, X_test, y_train, y_test = train_test_split(trained_results, y, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)         # Train model from data

    p_train = model.predict(X_train)    # Predict X_train after training
    p_test = model.predict(X_test)      # Predict X_test after training

    # We need to know the model mean squared error
    mse_train = round(mean_squared_error(y_train, p_train), 5)
    mse_test = round(mean_squared_error(y_test, p_test), 5)
    print('MSE train', mse_train)
    print('MSE test', mse_test)


for name in training_results.keys():
    r = training_results.get(name)
    print(f'#---- {name} ----#')
    print('Variables:', len(r.get('features_count')))
    learn(X_train_bigram_df)


exit(0)
