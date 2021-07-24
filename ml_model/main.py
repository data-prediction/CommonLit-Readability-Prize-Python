#!/usr/bin/env python3

import os

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


project_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(os.path.dirname(project_dir), 'input')
commonlitreadabilityprize_input_dir = os.path.join(input_dir, 'commonlitreadabilityprize')

with open(os.path.join(commonlitreadabilityprize_input_dir, 'train.csv')) as train_csv_fp:
    train_csv_df = pd.read_csv(train_csv_fp)

with open(os.path.join(commonlitreadabilityprize_input_dir, 'train.csv')) as test_csv_fp:
    test_csv_df = pd.read_csv(test_csv_fp)

# X = train_csv_df[['excerpt', 'standard_error']]
X = train_csv_df['excerpt']
y = train_csv_df['target'].values

vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


exit(0)

