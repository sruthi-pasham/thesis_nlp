from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score, precision_score
from merger import train, valid, test, y_train, y_valid, y_test
import numpy as np
import gensim
from gensim.models import Word2Vec

def unigram(data):
  unigrams = []
  for str in data:
    lst_words = str.split()
    lst_grams = [" ".join(lst_words[i:i+1]) 
                for i in range(0, len(lst_words), 1)]
    unigrams.append(lst_grams)
  return unigrams

len(unigram(test))

unigram_train = unigram(train)
unigram_valid = unigram(valid)
unigram_test = unigram(test)

training_corpus = unigram_train

print(np.asarray(training_corpus).shape)
#corpus[2]

#Word2Vec model (skipgram)
w2v = gensim.models.Word2Vec(training_corpus, min_count = 2,size = 100, window = 5, sg=1)

# Finding Word Vectors
print(w2v['python'].shape)

#testing
# Most similar words
similar = w2v.wv.most_similar('python')
print(similar)

#get vectors using above model, and pass it as parm to lr
print(train[0].split())

def compressor(partition):
  return [np.mean(w2v[[wrd for wrd in sent.split() if wrd in w2v.wv.vocab]], axis=0) for sent in partition]

x_train = compressor(train)
x_valid = compressor(valid)
x_test = compressor(test)

lr = OneVsRestClassifier(estimator=LogisticRegression()).fit(x_train, y_train)

preds_valid = lr.predict(x_valid)
#Accuracy
print(recall_score(y_valid, preds_valid, average='weighted'))
#Recall
print(recall_score(y_valid.T, preds_valid.T, average='macro'))
#Precision
print(precision_score(y_valid.T, preds_valid.T, average='macro'))
#F1
print(f1_score(y_valid.T, preds_valid.T, average='macro'))

preds_test = lr.predict(x_test)
#Accuracy
print(recall_score(y_test, preds_test, average='weighted'))
#Recall
print(recall_score(y_test.T, preds_test.T, average='macro'))
#Precision
print(precision_score(y_test.T, preds_test.T, average='macro'))
#F1
print(f1_score(y_test.T, preds_test.T, average='macro'))

