#!/usr/bin/env python3

import os

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

project_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(os.path.dirname(project_dir), 'input')
commonlitreadabilityprize_input_dir = os.path.join(input_dir, 'commonlitreadabilityprize')

with open(os.path.join(commonlitreadabilityprize_input_dir, 'train.csv')) as train_csv_fp:
    train_csv_df = pd.read_csv(train_csv_fp)

with open(os.path.join(commonlitreadabilityprize_input_dir, 'train.csv')) as test_csv_fp:
    test_csv_df = pd.read_csv(test_csv_fp)

# Show target statistics
sns.histplot(train_csv_df, x='target')
plt.show()



exit(0)





y = train_csv_df['target']
X = train_csv_df['excerpt']

countVectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
countVectorizer.fit(X)
X = countVectorizer.transform(X)

print(countVectorizer.get_feature_names())
print(X.todense())
print(X)
exit()
# X2 = countVectorizer.fit_transform(X)
# print(countVectorizer.get_feature_names())

X_train, X_test, y_train, y_test = train_test_split(X2, y, random_state=0)
model = BernoulliNB()
model.fit(X_train, y_train)

exit(0)
# X = train_csv_df[['excerpt', 'standard_error']]

print(train_csv_df.head())

y = train_csv_df['target']

model = BernoulliNB()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = mean_squared_error(y_train, p_train)
acc_test = mean_squared_error(y_test, p_test)

print(f'Train acc. {acc_train}; Test acc. {acc_test}')

exit(0)
