#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import nltk
import re
import operator
import sklearn


# In[2]:


path ='../datasets_coursework/Hateval/hateval.tsv'
dataset = pd.read_csv(path,sep='\t')  
dataset.head()


# In[3]:


combi = dataset


# In[ ]:


combi


# In[5]:


combi['text'] = combi['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
combi.head()


# In[35]:


tokenized_tweet = combi['text'].apply(lambda x: x.split())
tokenized_tweet.head()


# In[36]:


from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])


# In[37]:


combi['tidy_tweet'] = tokenized_tweet


# In[38]:


combi.head()


# In[39]:


dataset_full = []
for i in range(len(dataset)):
    dataset_full.append((combi['tidy_tweet'][i],dataset['label'][i]))


# In[12]:


import copy
lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")

# Function taken from Session 1
def get_list_tokens(string): # Function to retrieve the list of tokens from a string
    sentence_split=nltk.tokenize.sent_tokenize(string)
    list_tokens=[]
    for sentence in sentence_split:
        list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
        for token in list_tokens_sentence:
            list_tokens.append(lemmatizer.lemmatize(token).lower())
#         capitalizer = lambda x: ' '.join([w for w in x.split() if len(w)>3])
#         list_tokens = list(map(capitalizer, list_tokens))
    return list_tokens
    
# Function taken from Session 2
def get_vector_text(list_vocab,string):
    vector_text=np.zeros(len(list_vocab))
    
    string = re.sub("@[\w]*","", string)
    # remove special characters, numbers, punctuations
    string = string.replace("[^a-zA-Z#]", " ")
    
    list_tokens_string=get_list_tokens(string)
    for i, word in enumerate(list_vocab):
        if word in list_tokens_string:
            vector_text[i]=list_tokens_string.count(word)
    return vector_text


# Functions slightly modified from Session 2

def get_vocabulary(training_set, num_features): # Function to retrieve vocabulary
    dict_word_frequency={}
    for instance in training_set:
        sentence_tokens=get_list_tokens(instance[0])
        for word in sentence_tokens:
            if word in stopwords: continue
            if word not in dict_word_frequency: dict_word_frequency[word]=1
            else: dict_word_frequency[word]+=1
    sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_features]
    vocabulary=[]
    for word,frequency in sorted_list:
        vocabulary.append(word)
    return vocabulary

 

def train_svm_classifier(training_set, vocabulary): # Function for training our svm classifier
    X_train=[]
    Y_train=[]
    for instance in training_set:
        vector_instance=get_vector_text(vocabulary,instance[0])
        X_train.append(vector_instance)
        Y_train.append(instance[1])
    # Finally, we train the SVM classifier 
    svm_clf=sklearn.svm.SVC(kernel="linear",gamma='auto')
    svm_clf.fit(np.asarray(X_train),np.asarray(Y_train))
    return svm_clf

####regression


# In[13]:


from sklearn.model_selection import train_test_split
import random
size_dataset_full=len(dataset_full)
size_test=int(round(size_dataset_full*0.2,0))

list_test_indices=random.sample(range(size_dataset_full), size_test)
train_set=[]
test_set=[]      
for i,example in enumerate(dataset_full):
    if i in list_test_indices: test_set.append(example)
    else: train_set.append(example)


# In[23]:


print(size_dataset_full)
vocabulary=get_vocabulary(train_set, 1000)


# In[14]:


from sklearn.model_selection import KFold
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score


# In[15]:


dataset_full
kf = KFold(n_splits=10)
random.shuffle(dataset_full)
kf.get_n_splits(dataset_full)
for train_index, test_index in kf.split(dataset_full):
    train_set_fold=[]
    test_set_fold=[]
    accuracy_total=0.0
    for i,instance in enumerate(dataset_full):
        if i in train_index:
            train_set_fold.append(instance)
        else:
            test_set_fold.append(instance)
    vocabulary_fold=get_vocabulary(train_set_fold, 500)
    svm_clf_fold=train_svm_classifier(train_set_fold, vocabulary_fold)
    X_test_fold=[]
    Y_test_fold=[]
  
    for instance in test_set_fold:
        vector_instance=get_vector_text(vocabulary_fold,instance[0])
        X_test_fold.append(vector_instance)
        Y_test_fold.append(instance[1])
    Y_test_fold_gold=np.asarray(Y_test_fold)
    X_test_fold=np.asarray(X_test_fold)
    Y_test_predictions_fold=svm_clf_fold.predict(X_test_fold)
    accuracy_fold=accuracy_score(Y_test_fold_gold, Y_test_predictions_fold)
    accuracy_total+=accuracy_fold
    print ("Fold completed.")
average_accuracy=accuracy_total/5
print ("\nAverage Accuracy: "+str(round(accuracy_fold,3)))


# In[ ]:




