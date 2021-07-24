#!/usr/bin/env python3

import os
import re
import string

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize

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


# Show target statistics
sns.histplot(train_csv_df, x='target')
plt.show()


# Text processing

stopwords = set(STOPWORDS)
stopwords.update(stop_words_csv_df.word)

# Show Word Cloud

wordcloud = WordCloud(stopwords=stopwords).generate('.\n'.join(train_csv_df.excerpt))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

exit(0)


def normalize_text(text: str):
    print(text)
    text2 = word_tokenize(text)
    print(text2)
    return 'text'


train_csv_df['excerpt'] = train_csv_df['excerpt'].transform(
    lambda excerpt: normalize_text(excerpt), axis=0
)

exit(0)

excerpt_texts = ".\n".join(excerpt for excerpt in train_csv_df.excerpt)
excerpt_texts = re.sub(r'\d+', '', excerpt_texts)
excerpt_texts = excerpt_texts.translate(string.maketrans('', ''), string.punctuation)



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
