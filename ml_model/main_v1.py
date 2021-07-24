#!/usr/bin/env python3

import os
import re
import string

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from wordcloud import WordCloud, STOPWORDS
from sklearn.neural_network import MLPRegressor


# ---------------------------- Read external files --------------------------- #

project_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(os.path.dirname(project_dir), 'input')
commonlitreadabilityprize_input_dir = os.path.join(input_dir, 'commonlitreadabilityprize')
custom_input_dir = os.path.join(input_dir, 'custom')

with open(os.path.join(commonlitreadabilityprize_input_dir, 'train.csv')) as train_csv_fp:
    train_csv_df = pd.read_csv(train_csv_fp)

with open(os.path.join(commonlitreadabilityprize_input_dir, 'train.csv')) as test_csv_fp:
    test_csv_df = pd.read_csv(test_csv_fp)

with open(os.path.join(custom_input_dir, 'google_1gram_cnt.csv')) as stop_words_csv:
    stop_words_csv_df = pd.read_csv(stop_words_csv)


# -------------------------- Show target statistics -------------------------- #

sns.histplot(train_csv_df, x='target')
plt.show()


# ----------------------------- Text processing ------------------------------ #

stopwords = set(STOPWORDS)
stopwords.update(stop_words_csv_df.word)

# Show Word Cloud
wordcloud = WordCloud(stopwords=stopwords).generate('.\n'.join(train_csv_df.excerpt))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Vectorize text by occurrences
count_vector = CountVectorizer(ngram_range=(1, 2))
X_train_counts = count_vector.fit_transform(train_csv_df.excerpt)
print(X_train_counts.shape)
print(count_vector.vocabulary_.get(u'algorithm'))

# Transform text by frequencies
tf_transformer = TfidfTransformer(use_idf=False, smooth_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)

# Bigram (too many features)
bigram_vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    token_pattern=r'\b\w+\b', min_df=1
)
X_train_bigram = bigram_vectorizer.fit_transform(train_csv_df.excerpt)
print(X_train_bigram.shape)


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
    print(f'Model: {type(trained_results)}')
    print('MSE train', mse_train)
    print('MSE test', mse_test)


learn(X_train_counts)
learn(X_train_tf)
learn(X_train_bigram)

exit(0)
