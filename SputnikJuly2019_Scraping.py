#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:23:30 2019

@author: Alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:27:13 2019

@author: Alex
"""

#Lets See if we can scrape just one month of news articles from Sputnik

#Load in modules
from urllib.request import urlopen
from bs4 import BeautifulSoup
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
import string
import re
import preprocessor as p
%matplotlib inline

# Set up our scraper to only include July dates. 

# We will extract the date, title, and summary of the article
articles_sput_list=[]
for i in range(20190701, 20190730):
    url = 'https://sputniknews.com/archive/' + str(i)
    html = urlopen(url)
    soup = BeautifulSoup(html, 'lxml')
    sput_data = soup.find_all("div", {"class": "b-plainlist__info"})
    for item in sput_data:
        sput_date = item.contents[0].text
        sput_title = item.contents[1].text
        sput_overview = item.contents[2].text
        articles_sput=[sput_date, sput_title, sput_overview]
        articles_sput_list.append(articles_sput)

#Store into a dataframe
df_sput= pd.DataFrame(articles_sput_list)

#Rename our columns
columns = ['date', 'title', 'summary']
df_sput.columns = columns

#We can clean the files using some lambda functions
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

df_sput['title_clean'] = df_sput['title'].apply(lambda x: remove_punct(x))
df_sput.head(10)

#Tokenization
def tokenization(text):
    text = re.split('\W+', text)
    return text

df_sput['title_token'] = df_sput['title_clean'].apply(lambda x: tokenization(x.lower()))
df_sput.head()

#Stopwords
stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
df_sput['title_nonstop'] = df_sput['title_token'] .apply(lambda x: remove_stopwords(x))
df_sput.head(10)

#Stemming and Lemmitization
ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

df_sput['title_stem'] = df_sput['title_nonstop'].apply(lambda x: stemming(x))
df_sput.head()

wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

df_sput['title_lemm'] = df_sput['title_nonstop'].apply(lambda x: lemmatizer(x))
df_sput.head()

#Clean 
def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text

#Vectorization
countVectorizer = CountVectorizer(analyzer=clean_text) 
countVector = countVectorizer.fit_transform(df_sput['title'])
print('{} Number of tweets has {} words'.format(countVector.shape[0], countVector.shape[1]))
#print(countVectorizer.get_feature_names())
count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
count_vect_df.head()

#Join all words
text = " ".join(title for title in df_sput.summary)
print ("There are {} words in the combination of all review.".format(len(text)))

# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["Sputnik", "two", "new", "one", "said", "friday", "thursday", "wednesday"])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
