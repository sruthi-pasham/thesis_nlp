from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix
from merger import train_bl as train, valid_bl as valid, test_bl as test, y_train, y_valid, y_test, test as annots
from normalization import test_df
import numpy as np
import gensim
from gensim.models import Word2Vec

print(gensim.__version__)

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
w2v = gensim.models.Word2Vec(training_corpus, min_count = 2, size = 100, window = 5, sg=1)

# Finding Word Vectors


#testing
# Most similar words
similar = w2v.wv.most_similar('python')
print(similar)

#get vectors using above model, and pass it as parm to lr
print(train[0].split())

def compressor(partition):
  return np.array([np.mean(w2v[[wrd for wrd in sent.split() if wrd in w2v.wv.vocab]], axis=0) for sent in partition])

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
print(recall_score(y_test.T, preds_test.T, average='macro', ))
#Precision
print(precision_score(y_test.T, preds_test.T, average='macro'))
#F1
print(f1_score(y_test.T, preds_test.T, average='macro'))


unique_jobs =  test_df['RESULT'].unique()
def jobs_first():
    first_occ = []
    df = test_df.reset_index()#.to_numpy()
    for j in unique_jobs:
        first_occ.append(df.loc[df['RESULT'] == j].index[0])
    return first_occ

first_occ = jobs_first()
#returns an array woth the number of FPs for each job title


def c_matrix1(test, lst, preds):
    f_arr = []
    for tup in [(0,0),(0,1),(1,0),(1,1)]:
        helper_arr = []
        for i, p in enumerate(preds):
            helper_arr.append(confusion_matrix(test[i], p)[tup[0]][tup[1]])
        for num in range(len(lst)-1): 
            f_arr.append(round(sum(helper_arr[lst[num]:lst[num+1]])/len(helper_arr[lst[num]:lst[num+1]]), 2))
            if num+1== len(lst)-1:
                f_arr.append(round(sum(helper_arr[lst[num+1]:])/len(helper_arr[lst[num+1]:]),2))
    return np.array(f_arr).reshape((4,30))

cm = c_matrix1(y_test, first_occ, preds_test)

print(cm[1])
print(np.average(cm[1]))
#print(metrics[3])

def c_matrix2(test, preds):
    arr = []
    removed_courses = []
    for tup in [(0,0),(0,1),(1,0),(1,1)]:
        for i, p in enumerate(preds):
            if np.sum(p) + np.sum(test[i]) == 0:
                removed_courses.append(i)
                continue
            arr.append(confusion_matrix(test[i], p)[tup[0]][tup[1]])
    return np.array(arr).reshape((4,46)), removed_courses

metrics, rc = c_matrix2(y_test.T, preds_test.T)

#print(rc)

print(metrics)

def balanced_accuracy():
    ratio = []
    for i in range(46):
        tpr = (metrics[3][i])/(metrics[2][i]+metrics[3][i]) 
        tnr = (metrics[0][i])/(metrics[1][i]+metrics[0][i])
        ratio.append(round((tpr+tnr)/2,2))
    return ratio

#print(unique_jobs)

ba = balanced_accuracy()
print(balanced_accuracy())
print(np.average(ba))

#tn, fp, fn, tp

#courses counts
counts_courses = np.sum(preds_test, axis=0)

#which courses appear the most, right or wrong
#positive likelihood ratio = (TP/FN+TP)/(FP/TN+FP)

#% of courses covered by each job title
def coverage(y_test, tp):
    return [round(tp[i]/np.sum(y_test[i]),2) for i in range(len(y_test))]

print(coverage(annots, cm[3]))

