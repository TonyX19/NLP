
#coding:utf-8
import sys
import keras
 
 
 
 
VECTOR_DIR = 'vectors.bin'
 
 
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128
VALIDATION_SPLIT = 0.16
TEST_SPLIT = 0.2
 
 
 
 
print ('(1) load texts...')
train_texts = open('train_contents.txt',encoding='utf-8').read().split('\n')
train_labels = open('train_labels.txt',encoding='utf-8').read().split('\n')
test_texts = open('test_contents.txt',encoding='utf-8').read().split('\n')
test_labels = open('test_labels.txt',encoding='utf-8').read().split('\n')
all_texts = train_texts + test_texts
all_labels = train_labels + test_labels
 
 
 
 
print ('(2) doc to var...')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
 
 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)
sequences = tokenizer.texts_to_sequences(all_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(all_labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
 
 
 
 
print ('(3) split data set...')
p1 = int(len(data)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(data)*(1-TEST_SPLIT))
x_train = data[:p1]
y_train = labels[:p1]
x_val = data[p1:p2]
y_val = labels[p1:p2]
x_test = data[p2:]
y_test = labels[p2:]
print ('train docs: '+str(len(x_train)))
print ('val docs: '+str(len(x_val)))
print ('test docs: '+str(len(x_test)))
 
 
 
 
print ('(4) load word2vec as embedding...')
import gensim
from keras.utils import plot_model
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True)
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
not_in_model = 0
in_model = 0
for word, i in word_index.items(): 
    if word in w2v_model:
        in_model += 1
        embedding_matrix[i] = np.asarray(w2v_model[word], dtype='float32')
    else:
        not_in_model += 1
print (str(not_in_model)+' words not in w2v model')
from keras.layers import Embedding
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
 
 
 
 
print ('(5) training model...')
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import LSTM, Embedding
from keras.models import Sequential
 
 
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.2))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()
#plot_model(model, to_file='model.png',show_shapes=True)
 
 
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
print (model.metrics_names)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=128)
#model.save('word_vector_lstm.h5')
 
 
print ('(6) testing model...')
print (model.evaluate(x_test, y_test))
 
 
        
#画图
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import numpy as np
from scipy import interp
 
 
y_score  = model.predict(x_test)
lw = 2
n_classes = 11
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
 
 
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
 
 
 
 
# Compute macro-average ROC curve and ROC area
 
 
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
 
 
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
 
 
# Finally average it and compute AUC
mean_tpr /= n_classes
 
 
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
 
 
 
 
 
 
# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
 
 
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
 
 
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
 
 
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
