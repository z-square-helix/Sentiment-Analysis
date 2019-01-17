### TWITTER SENTIMENT ANALYSIS PROJECT ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv("./trainingandtestdata/training.1600000.processed.noemoticon.csv", header = None, names = cols, encoding = "ISO-8859-1")

# print(df.head())

# print(df.sentiment.value_counts())

df.drop(['id','date','query_string','user'], axis = 1, inplace = True)

# PRINTING SOME ROWS OF EACH TO GET INDEX VALUES

# print(df[df.sentiment == 0].head(10))

# print(df[df.sentiment == 4].head(10))

# NEGATIVE (0) 0 - 799999
# POSITIVE (4) 800000 - 1600000

# LENGTH OF STRING IN TEXT COLUMN

df['sentiment'] = df['sentiment'].map({0:0, 4:1})

df['not_clean_len'] = [len(t) for t in df.text]

from pprint import pprint

data_dict = {
    'sentiment':{
        'type':df.sentiment.dtype,
        'description':'sentiment class - 0:negative, 1:positive'
    },
    'text':{
        'type':df.text.dtype,
        'description':'tweet text'
    },
    'not_clean_len':{
        'type':df.not_clean_len.dtype,
        'description':'tweet length before cleaning'
    },
    'dataset_shape':df.shape
}

# pprint(data_dict)

# VISUALIZING DISTRIBUTION OF LENGTH

fig, ax = plt.subplots(figsize = (5, 5))
plt.boxplot(df.not_clean_len)
# plt.show()

# CLEANING AND TOKENIZING
# REMOVING HASHTAGS, @ MENTIONS, AND URLS
# DECODING HTML

import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()

www_pat = r'www.[^ ]+'
negative_dict = {"isn't" : "is not", "aren't" : "are not", "wasn't" : "was not", "weren't" : "were not", "haven't" : "have not", "hasn't" : "has not", "hadn't" : "had not", "won't" : "will not", "wouldn't" : "would not", "don't" : "do not", "doesn't" : "does not", "didn't" : "did not", "can't" : "can not", "couldn't" : "could not", "shouldn't" : "should not", "mightn't" : "might not", "mustn't" : "must not"}
neg_pat = re.compile(r'\b(' + '|'.join(negative_dict.keys()) + r')\b')

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()

    try:
        remove = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        remove = souped

    stripped = re.sub(r'@[A-Za-z0-9]+', '', remove)
    stripped = re.sub('https?://[A-Za-z0-9./]+', '', stripped)
    stripped = re.sub(www_pat, '', stripped)

    lower = stripped.lower()

    neg_replace = neg_pat.sub(lambda x: negative_dict[x.group()], lower)

    letters_only = re.sub("[^a-zA-Z]", " ", neg_replace)

    words = [x for x in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()

# test = df.text[:100]

# test_result = []
# for t in test:
#     test_result.append(tweet_cleaner(t))
# print(test_result)

print('Cleaning and parsing \n')
clean_tweet_text = []
for x in range(0, len(df)):
    if( ( x + 1 )%10000 == 0):
        print('%d of %d have been processed' % ( x+1, len(df) ))
    clean_tweet_text.append(tweet_cleaner(df['text'][x]))

# SAVING AS CSV

clean_df = pd.DataFrame(clean_tweet_text, columns = ['text'])
clean_df['target'] = df.sentiment
# print(clean_df.head())

clean_df.to_csv('clean_tweets.csv', encoding='utf-8')

csv = 'clean_tweets.csv'
my_df = pd.read_csv(csv, index_col=0)
# print(my_df.head())

# print(my_df.info())
# print(my_df[my_df.isnull().any(axis=1)].head())
# print(np.sum(my_df.isnull().any(axis=1)))
    # >>> 3,981 ENTRIES HAVE NULL TEXT AFTER CLEANING

# DROPPING NULL ENTRIES

my_df.dropna(inplace = True)
my_df.reset_index(drop = True, inplace = True)
# print(my_df.info())

# GETTING FREQUENCY DATA

from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer()
cvec.fit(my_df.text)

print(len(cvec.get_feature_names()))

neg_doc_matrix = cvec.transform(my_df[my_df.target == 0].text)
pos_doc_matrix = cvec.transform(my_df[my_df.target == 1].text)
neg_tf = np.sum(neg_doc_matrix, axis = 0)
pos_tf = np.sum(pos_doc_matrix, axis = 0)
neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))
term_freq_df = pd.DataFrame([neg, pos], columns = cvec.get_feature_names()).transpose()

term_freq_df.columns = ['negative', 'positive']
term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
sorted_clean = term_freq_df.sort_values(by='total', ascending=False).iloc[:10]

print(sorted_clean)

term_freq_df.to_csv('term_freq_df.csv', encoding = 'utf-8')

term_freq_df = pd.read_csv('term_freq_df.csv', index_col = 0, encoding = 'utf-8')

cvec_two = CountVectorizer(stop_words = 'english', max_features = 10000)
cvec_two.fit(my_df.text)

document_matrix = cvec.transform(my_df.text)

neg_batches = np.linspace(0, 798179, 10).astype(int)
x = 0
neg_tf = []
while x < len(neg_batches) - 1:
    batch_result = np.sum(document_matrix[neg_batches[x]:neg_batches[x+1]].toarray(), axis = 0)
    neg_tf.append(batch_result)
    print(neg_batches[x+1], "entries term freq calculated")
    x += 1

pos_batches = np.linspace(798179, 1596019, 10).astype(int)
x = 0
pos_tf = []
while x < len(pos_batches) - 1:
    batch_result = np.sum(document_matrix[pos_batches[x]:pos_batches[x+1]].toarray(), axis = 0)
    pos_tf.append(batch_result)
    print(pos_batches[x+1], "entries term freq calculated")
    x += 1

neg = np.sum(neg_tf, axis = 0)
pos = np.sum(pos_tf, axis = 0)
term_freq_df2 = pd.DataFrame([neg,pos], columns = cvec.get_feature_names()).transpose()
term_freq_df2.columns = ['negative', 'positive']
term_freq_df2['total'] = term_freq_df2['negative'] + term_freq_df2['positive']
sorted2 = term_freq_df2.sort_values(by = 'total', ascending = False).iloc[:10]
print(sorted2)

# RATE A POSITIVE WORD APPEARS

term_freq_df2['pos_rate'] = term_freq_df2['positive'] * 1./term_freq_df2['total']

# FREQUENCY OF WORD IN CLASS

term_freq_df2['pos_freq_pct'] = term_freq_df2['positive'] * 1./term_freq_df2['positive'].sum()

# CALCULATING THE CUMULATIVE DISTRIBUTION FUNCITON
# OF POS_RATE AND POS_FREQ, THEN TAKING THE HARMONIC
# MEAN OF BOTH CDF SO AS TO COUNT THE SMALLER FREQ
# VALUE WITHOUT SO MUCH BIAS TOWARD RATE

from scipy.stats import hmean, norm
def normcdf(x):
    return norm.cdf(x, x.mean(), x.std())

term_freq_df2['pos_rate_normcdf'] = normcdf(term_freq_df2['pos_rate'])
term_freq_df2['pos_freq_normcdf'] = normcdf(term_freq_df2['pos_freq_pct'])
term_freq_df2['pos_normcdf_hmean'] = hmean([term_freq_df2['pos_rate_normcdf'], term_freq_df2['pos_freq_normcdf']])
term_freq_df2.sort_values(by = 'pos_normcdf_hmean', ascending = False).iloc[:10]

# REPEATING FOR NEGATIVE WORDS

term_freq_df2['neg_rate'] = term_freq_df2['negative'] * 1./term_freq_df2['total']

term_freq_df2['neg_freq_pct'] = term_freq_df2['negative'] * 1./term_freq_df2['negative'].sum()

term_freq_df2['neg_rate_normcdf'] = normcdf(term_freq_df2['neg_rate'])
term_freq_df2['neg_freq_normcdf'] = normcdf(term_freq_df2['neg_freq_pct'])
term_freq_df2['neg_normcdf_hmean'] = hmean([term_freq_df2['neg_rate_normcdf'], term_freq_df2['neg_freq_normcdf']])
term_freq_df2.sort_values(by = 'neg_normcdf_hmean', ascending = False).iloc[:10]

# VISUALIZING CDF HARMONIC MEANS

plt.figure(figsize = (8, 6))
ax = sns.regplot(x = "neg_normcdf_hmean", y = "pos_normcdf_hmean", fit_reg = False, scatter_kws = {'alpha' : 0.5}, data = term_freq_df2)
plt.ylabel('Harmonic Mean of Positive Rate CDF and Frequency CDF')
plt.xlabel('Harmonic Mean of Negative Rate CDF and Frequency CDF')
plt.title('neg_normcdf_hmean vs pos_normcdf_hmean')
