#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:42:48 2019

@author: Alex
"""
############################################
#Cleaning Messy Text Data: Useful Functions#
############################################

#Code from https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/

#Load packages
from nltk.corpus import stopwords

#Load in text data from Sputnik, a Russian internatioanl media network, for (August 2019)
sputnik = pd.read_csv('sputnikaugustdata.csv', encoding = 'ISO-8859-1', index_col=0)

#Examine data
sputnik.head()
sputnik.columns

#Lets focus on the summary columns

# Function to remove non-ASCII
def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i)<128)
 
sputnik['summary'] = sputnik['summary'].apply(remove_non_ascii)
sputnik['summary'].head()

#Get number of words first
sputnik['word_count'] = sputnik['summary'].apply(lambda x: len(str(x).split(" ")))
sputnik[['summary','word_count']].head()

#Get the average word length
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

sputnik['avg_word'] = sputnik['summary'].apply(lambda x: avg_word(x))
sputnik[['summary','avg_word']].head()

#Number of stop words
stop = stopwords.words('english')

sputnik['stopwords'] = sputnik['summary'].apply(lambda x: len([x for x in x.split() if x in stop]))
sputnik[['summary','stopwords']].head()

# Number of unique words
sputnik['num_unique_words'] = sputnik['summary'].apply(lambda x: len(set(w for w in x.split())))
sputnik[['summary','num_unique_words']].head()

#Ratio of words vs unique words
sputnik['words_vs_unique'] = sputnik['num_unique_words'] / sputnik['word_count']
sputnik[['summary','words_vs_unique']].head()

#Number of numerics
sputnik['numerics'] = sputnik['summary'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
sputnik[['summary','numerics']].head()

#Number of upper case
train['upper'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
train[['tweet','upper']].head()

#Get the parts of Speech
def tag_part_of_speech(text):
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    pos_list = pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    adjective_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    verb_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return[noun_count, adjective_count, verb_count]
    
sputnik['nouns'], sputnik['adjectives'], sputnik['verbs'] = zip(*sputnik['summary'].apply(lambda comment: tag_part_of_speech(comment)))
sputnik['nouns_vs_words'] = sputnik['nouns'] / sputnik['word_count']
sputnik['adjectives_vs_words'] = sputnik['adjectives'] / sputnik['word_count']
sputnik['verbs_vs_words'] = sputnik['verbs'] / sputnik['word_count']

# Lower case 
sputnik['summary'] = sputnik['summary'].apply(lambda x: " ".join(x.lower() for x in x.split()))
sputnik['summary'].head()

# Remove Punctuation
sputnik['summary'] = sputnik['summary'].str.replace('[^\w\s]','')
sputnik['summary'].head()

#Remove stop words
sputnik['summary'] = sputnik['summary'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
sputnik['summary'].head()

#Can also remove most common words (10 here)
freq = pd.Series(' '.join(sputnik['summary']).split()).value_counts()[:10]
freq

freq = list(freq.index)
sputnik['summary'] = sputnik['summary'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
sputnik['summary'].head()

#Can also remove the least common words (10 Here)
freq = pd.Series(' '.join(sputnik['summary']).split()).value_counts()[-10:]
freq

freq = list(freq.index)
sputnik['summary'] = sputnik['summary'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
sputnik['summary'].head()

#Spelling Correction (This takes a lot of time!)
from textblob import TextBlob
sputnik['summary'][:5].apply(lambda x: str(TextBlob(x).correct()))  # only doing first 5 rows

#Stemming the words
from nltk.stem import PorterStemmer
st = PorterStemmer()
sputnik['summary_stem']= sputnik['summary'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

#Often we prefer lemmatization
from textblob import Word
sputnik['summary_lemm'] = sputnik['summary'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
sputnik['summary_lemm'].head()

#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
stop_words= 'english',ngram_range=(1,1))
train_vect = tfidf.fit_transform(sputnik['summary_lemm'])
train_vect

# TF-IDF by Scikit learn.
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1, max_df=50)
X = vectorizer.fit_transform(sputnik['summary_lemm'])
 

#Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(sputnik['summary_lemm'])
train_bow

#Textblob sentiment analysis
sputnik['sentiment'] = sputnik['summary_lemm'].apply(lambda x: TextBlob(x).sentiment[0])
sputnik[['summary_lemm','sentiment']].head()

