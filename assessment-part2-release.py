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


import nltk
from nltk import FreqDist
#nltk.download('stopwords') # run this one time

import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
import re
import spacy


# In[3]:


import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


pos_path ='./datasets_coursework/IMDb/train/imdb_train_pos.txt'
neg_path ='./datasets_coursework/IMDb/train/imdb_train_neg.txt'
dataset_train_pos = pd.read_csv(pos_path,sep='\n',header=None)
dataset_train_neg = pd.read_csv(neg_path,sep='\n',header=None) 
training_neg_set = dataset_train_neg[0]
training_pos_set = dataset_train_pos[0]

print(dataset_train_neg.shape[0],dataset_train_neg.shape[1])
#二维数组长宽


# In[5]:


pos_path ='./datasets_coursework/IMDb/dev/imdb_dev_pos.txt'
neg_path ='./datasets_coursework/IMDb/dev/imdb_dev_neg.txt'
dataset_dev_pos = pd.read_csv(pos_path,sep='\n',header=None)
dataset_dev_neg = pd.read_csv(neg_path,sep='\n',header=None) 
dev_neg_set = dataset_dev_neg[0]
dev_pos_set = dataset_dev_pos[0]


# In[6]:


pos_path ='./datasets_coursework/IMDb/test/imdb_test_pos.txt'
neg_path ='./datasets_coursework/IMDb/test/imdb_test_neg.txt'
dataset_test_pos = pd.read_csv(pos_path,sep='\n',header=None)
dataset_test_neg = pd.read_csv(neg_path,sep='\n',header=None) 
test_neg_set = dataset_dev_neg[0]
test_pos_set = dataset_dev_pos[0]


# In[192]:


# function to plot most frequent terms
def freq_words(x, terms = 30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
    return words_df
#     # selecting top 20 most frequent words
#     d = words_df.nlargest(columns="count", n = terms) 
#     plt.figure(figsize=(20,5))
#     ax = sns.barplot(data=d, x= "word", y = "count")
#     ax.set(ylabel = 'Count')
#     plt.show()


# In[184]:


total_train_set = np.concatenate((dataset_train_pos[0], dataset_train_neg[0]), axis=0)


# In[188]:


total_train_new = []
for item in total_set_new:
    total_train_new.append(" ".join(item))


# In[193]:


a = freq_words(total_train_new,50)


# In[199]:


b = a.sort_values(by=['count'],ascending=False)


# In[7]:


#data preprocessing
from nltk.corpus import stopwords
#nltk.download('punkt')
#分词模型
#nltk.download('wordnet')
#同义词 lemma form lemmatization 词形还原

stopwords = set(stopwords.words('english'))

stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")

# function to remove stopwords
def remove_stopwords(rev): 
    #rev -> token_list
    output = []
    for i in rev:
        if i not in stopwords:
            output.append(i)
    return output




# # remove short words (length < 3)
# df = df.apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
# # remove stopwords from the text
# reviews = [remove_stopwords(r.split()) for r in df]
# #make entire text lowercase lemma form
# reviews = [lemmatizer.lemmatize(r).lower() for r in reviews]


# In[8]:


#!python -m spacy download en # one time run
nlp = spacy.load('en', disable=['parser', 'ner'])
def lemmatization(sent, tags=['NOUN', 'ADJ','ADV']): # filter noun and adjective
    #sent -> string
    output = []
    doc = nlp(sent) 
    for token in doc:
        if token.pos_ in tags:
            output.append(token.lemma_)
    return output


# In[9]:


lemmatizer = nltk.stem.WordNetLemmatizer()
# Function taken from Session 1
def get_list_tokens(string): # Function to retrieve the list of tokens from a string
    list_tokens=[]
    sentence_split = nltk.tokenize.sent_tokenize(string)   
    for sentence in sentence_split:
        list_tokens_sentence=remove_stopwords(lemmatization(sentence))
        for token in list_tokens_sentence:
            # remove short words (length < 3)
            if len(token) < 3 :
                continue;
            token_lemma = lemmatizer.lemmatize(token).lower()
            if  token_lemma == '-pron-':
                continue
            list_tokens.append(token_lemma)
    return list_tokens


# In[10]:


def get_vocabulary(training_set, num_features): # Function to retrieve vocabulary
    dict_word_frequency={}
    for sentence in training_set:
        sentence_tokens = get_list_tokens(sentence)
        for word in sentence_tokens:
            if word not in dict_word_frequency: dict_word_frequency[word]=1
            else: dict_word_frequency[word]+=1
    
    sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_features]
    vocabulary=[]
    for word,frequency in sorted_list:
        vocabulary.append(word)
    return vocabulary


# In[11]:


#vacabulary = get_vocabulary(df,100)
#print(dataset_train_pos)
#print(vacabulary)
vacabulary = ['good','well','bad','awful','old','favorite']


# In[12]:


# Function taken from Session 2
def get_vector_text(list_vocab,text):
    vector_text=np.zeros(len(list_vocab))
    if type(text) is str:
        list_tokens_string=get_list_tokens(string)
    for i, word in enumerate(list_vocab):
        if word in list_tokens_string:
            vector_text[i]=list_tokens_string.count(word)
    return vector_text


# In[13]:


def train_svm_classifier(training_set, vacabulary): # Function for training our svm classifier
    print(vacabulary)
    X_train=[]
    Y_train=[]
    for instance in training_set:
        vector_instance=get_vector_text(vacabulary,instance)
        X_train.append(vector_instance)
        Y_train.append(1)
        print(Y_train)
    # Finally, we train the SVM classifier 
    #svm_clf=sklearn.svm.SVC(kernel="linear",gamma='auto')
    #svm_clf.fit(np.asarray(X_train),np.asarray(Y_train))
    #return svm_clf


# In[14]:


total_set = np.concatenate((dataset_train_pos[0], dataset_train_neg[0]), axis=0)


# In[15]:


total_set_new = []

for item in total_set:
    total_set_new.append(get_list_tokens(item))


# In[16]:


import gensim
from gensim import corpora
dictionary = corpora.Dictionary(total_set_new) 


# In[206]:


topic_num = 10


# In[207]:


doc_term_matrix = [dictionary.doc2bow(rev) for rev in total_set_new]
# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix,
                                   id2word=dictionary,
                                   num_topics=topic_num, 
                                   random_state=100,
                                   chunksize=1000,
                                   passes=50)


# In[208]:


lda_model.print_topics() 


# In[47]:


topic_id = 0
topic_size = 7
topic_dict = [[]] * topic_size
for topic_id in range(topic_size):
    terms_id_list = lda_model.get_topic_terms(topic_id)
    for terms_id in terms_id_list:
        topic_dict[topic_id].append(dictionary[terms_id[0]])


# In[49]:


#topic_dict


# In[22]:


total_set_new[1]


# In[23]:


doc_bow = dictionary.doc2bow(total_set_new[1]) 
a = lda_model[doc_bow]


# In[24]:


def addTopicFeaAssemble(text_token_list):
    ret = []
    topic_fea = [0] * topic_num
    doc_bow = dictionary.doc2bow(text_token_list) 
    topic_weights = lda_model[doc_bow]
    for t in topic_weights:
        idx,weight = t
        topic_fea[idx] += weight  
            
    return topic_fea


# In[25]:


from gensim.models import word2vec


# In[26]:


def model(review_vec_list):
    model = word2vec.Word2Vec(review_vec_list,sg=0,hs=1,min_count=1,window=5,size=300)
    return model


# In[27]:


#training_set = np.concatenate((training_pos_set, training_neg_set), axis=0)


# In[28]:


# def transToVecList(dataset):
#     ret = []
#     i = 0
#     for item in dataset:
#         print(i)
#         ret.append(get_list_tokens(item))
#         i+=1
#     return ret


# In[29]:


#train_set_list = transToVecList(total_set_new)


# In[30]:


import time
print(time.time())
x_model = model(total_set_new)
print(time.time())


# In[172]:


def get_sent_vec(size,sent,model):
    vec = np.zeros(size).reshape(1,size)
    count = 0
    vec_list = []
#    length = len(sent)
    for topic_g in topic_dict:
        for word in sent:
            if word in topic_g:
                try:
                    vec += model.wv[word].reshape(1,size)
                    count += 1
                except:
                    continue
            if count != 0:
                vec /= count
        vec_list.append(sum(vec[0]))
    return [vec_list]
#     #topic_feas = np.asarray([addTopicFeaAssemble(sent)])
#     len_feas = np.asarray([[length]])
#     return np.hstack((vec,len_feas))
#     return np.hstack((vec,topic_feas,len_feas))


# In[173]:


def get_vec(train,model,size):
    word_vec = np.concatenate([get_sent_vec(size,sent,model) for sent in train])
    return word_vec


# In[ ]:





# In[174]:


train_vec = get_vec(total_set_new,x_model,300)


# In[176]:


print(train_vec[0])


# In[177]:


Y_train = [1] * len(dataset_train_pos[0])
Y_train += [0] * len(dataset_train_neg[0])


# In[141]:


svm_clf=sklearn.svm.SVC(kernel="linear",gamma='auto')
svm_clf.fit(np.asarray(train_vec),np.asarray(Y_train))


# In[149]:


svm_clf_1=sklearn.svm.SVC(kernel="linear",gamma='auto')
svm_clf_1.fit(np.asarray(train_vec),np.asarray(Y_train))


# In[178]:


svm_clf_2=sklearn.svm.SVC(kernel="linear",gamma='auto')
svm_clf_2.fit(np.asarray(train_vec),np.asarray(Y_train))


# In[179]:


# X_dev = []
# # Y_dev = []

# total_dev_set = np.concatenate((dev_pos_set, dev_neg_set), axis=0)
# for item in total_dev_set:
#     X_dev.append(get_list_tokens(item))
    
# get_vec(total_set_new,x_model,300)


# In[ ]:





# In[180]:


X_dev_vec = get_vec(X_dev,x_model,300)
Y_dev = [1] * len(dev_pos_set) + [0] * len(dev_neg_set)
X_dev_vec[2]


# In[181]:


y_dev_pre = svm_clf_2.predict(X_dev_vec)
from sklearn.metrics import classification_report
print(classification_report(Y_dev, y_dev_pre))


# In[151]:


y_dev_pre = svm_clf_1.predict(X_dev_vec)


# In[152]:


from sklearn.metrics import classification_report
print(classification_report(Y_dev, y_dev_pre))


# In[143]:


y_dev_pre = svm_clf.predict(X_dev_vec)


# In[144]:


from sklearn.metrics import classification_report
print(classification_report(Y_dev, y_dev_pre))


# In[ ]:


X_test = []
# Y_dev = []

total_dev_set = np.concatenate((test_pos_set, test_neg_set), axis=0)
for item in total_dev_set:
    X_dev.append(get_list_tokens(item))
    
get_vec(total_set_new,x_model,300)

