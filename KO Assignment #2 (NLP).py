#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
import re
import unicodedata
import inflect
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score


# # Importing Data

# In[4]:


df = pd.read_json('C:\\Users\\Omar Atef\\Desktop\\Knowledge Officer\\Datasets\\aggressive_dedup.json', lines=True, nrows=100000)


# # Data Exploration

# In[5]:


df.columns


# In[6]:


data = df[['reviewText','summary','overall']]
data.head()


# In[7]:


r = 99
print(data['reviewText'][r])
print('\n')
print(data['summary'][r])
print('\n')
print(data['overall'][r])


# In[8]:


data.describe()


# In[13]:


data['overall'].value_counts().sum()


# # Data Balancing

# In[12]:


data['overall'] = data['overall'].replace({5: 'good', 4: 'good', 3: 'neutral', 2: 'bad', 1: 'bad'})
data.head()


# In[14]:


data['overall'].value_counts().sum()


# In[15]:


data['overall'].value_counts()


# In[17]:


min_feature_count = data['overall'].value_counts().min()
min_feature_count

data_good = data[data['overall'] == 'good']
data_neutral = data[data['overall'] == 'neutral']
data_bad = data[data['overall'] == 'bad']

balanced_data_good = data_good.head(len(data_good) - (len(data_good) - min_feature_count))
balanced_data_neutral = data_neutral.head(len(data_neutral) - (len(data_neutral) - min_feature_count))
balanced_data_bad = data_bad.head(len(data_bad) - (len(data_bad) - min_feature_count))

print(len(balanced_data_good))
print(len(balanced_data_neutral))
print(len(balanced_data_bad))


# In[18]:


balanced_data = pd.concat([balanced_data_good,balanced_data_neutral,balanced_data_bad])
balanced_data = shuffle(balanced_data)
balanced_data = balanced_data.reset_index(drop=True)

sns.countplot('overall', data = balanced_data)


# In[ ]:


# min_feature_count = data['overall'].value_counts().min()
# min_feature_count

# data_1 = data[data['overall'] == 1]
# data_2 = data[data['overall'] == 2]
# data_3 = data[data['overall'] == 3]
# data_4 = data[data['overall'] == 4]
# data_5 = data[data['overall'] == 5]

# balanced_data_1 = data_1.head(len(data_1) - (len(data_1) - min_feature_count))
# balanced_data_2 = data_2.head(len(data_2) - (len(data_2) - min_feature_count))
# balanced_data_3 = data_3.head(len(data_3) - (len(data_3) - min_feature_count))
# balanced_data_4 = data_4.head(len(data_4) - (len(data_4) - min_feature_count))
# balanced_data_5 = data_5.head(len(data_5) - (len(data_5) - min_feature_count))

# print(len(balanced_data_1))
# print(len(balanced_data_2))
# print(len(balanced_data_3))
# print(len(balanced_data_4))
# print(len(balanced_data_5))


# In[ ]:


# balanced_data = pd.concat([balanced_data_1,balanced_data_2,balanced_data_3,balanced_data_4,balanced_data_5])
# balanced_data = shuffle(balanced_data)
# balanced_data = balanced_data.reset_index(drop=True)

# sns.countplot('overall', data = balanced_data)


# In[19]:


balanced_data.head(10)


# In[21]:


balanced_data['overall'] = balanced_data['overall'].replace({'good':0, 'neutral':1, 'bad':2})
balanced_data.head(10)


# In[22]:


text_data = balanced_data[['reviewText','overall']]

summary_data = balanced_data[['summary','overall']]


# # Data Cleaning and Prepocessing

# In[23]:


def remove_non_ascii(reviewText):
    new_words = []
    for word in reviewText:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(reviewText):
    new_words = []
    for word in reviewText:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(reviewText):
    new_words = []
    for word in reviewText:
        new_word = re.sub(r'[^\w\s]', '', word)
        newest_word = re.sub('btw', 'by the way', new_word)
        new = re.sub('_', '', new_word)
        clean_word = re.sub('@[^ ]+', '', new)
        cleaner = re.sub("'ve", ' have', clean_word)
        if cleaner != '':
            new_words.append(cleaner)
    return new_words

def replace_numbers(reviewText):
    p = inflect.engine()
    new_words = []
    for word in reviewText:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def lemmatize_verbs(reviewText):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in reviewText:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def remove_stopwords(reviewText):
    new_words = []
    for word in reviewText:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def normalize(reviewText):
    reviewText = remove_non_ascii(reviewText)
    reviewText = to_lowercase(reviewText)
    reviewText = remove_punctuation(reviewText)
    #reviewText = replace_numbers(reviewText)
    #reviewText = remove_stopwords(reviewText)
    reviewText = lemmatize_verbs(reviewText)
    return reviewText


# In[24]:


text_data['text'] = text_data['reviewText'].apply(lambda x:' '.join(normalize(x.split())))
text_data


# In[25]:


features = text_data['text']
targets = text_data['overall']


# # Train-Test Split

# In[26]:


train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.3)


# In[27]:


cv = CountVectorizer(stop_words='english', analyzer='word', min_df=2)


# # Feature Transformation

# In[28]:


features_cv = cv.fit_transform(features)


# In[29]:


features_cv.toarray()


# In[30]:


tfidf = TfidfTransformer()
features_tfidf = tfidf.fit_transform(features_cv)
features_tfidf.shape


# In[31]:


train_features, test_features, train_targets, test_targets = train_test_split(features_cv, targets, test_size=0.3)


# # Model Training and Testing

# In[32]:


nb = MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None).fit(train_features, train_targets)
nb.predict(test_features)
print(nb.score(train_features, train_targets))
print(nb.score(test_features, test_targets))


# In[33]:


bc = BaggingClassifier().fit(train_features, train_targets)
bc.predict(test_features)
print(bc.score(train_features, train_targets))
print(bc.score(test_features, test_targets))


# In[34]:


svc = LinearSVC(penalty='l2', max_iter=10000).fit(train_features, train_targets)
svc.predict(test_features)
print(svc.score(train_features, train_targets))
print(svc.score(test_features, test_targets))


# In[49]:


lr = LogisticRegression(penalty='l2', tol=0.0001, C=1, multi_class='multinomial', max_iter=5000).fit(train_features, train_targets)


# In[50]:


val_scores = cross_val_score(lr, train_features, train_targets, cv=5, scoring="accuracy")


# In[51]:


lr.predict(test_features)


# In[52]:


lr.score(train_features, train_targets)


# In[53]:


val_scores


# In[55]:


lr.score(test_features, test_targets)

