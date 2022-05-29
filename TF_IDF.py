from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix
from merger import train_bl as train, valid_bl as valid, test_bl as test, y_train, y_valid, y_test, test as annots
from normalization import test_df
import pandas as pd
from courses import courses
import numpy as np

vectorizer = TfidfVectorizer(max_features=2000)
x_train = vectorizer.fit_transform(train).toarray()
x_valid = vectorizer.fit_transform(valid).toarray()
x_test = vectorizer.fit_transform(test).toarray()

print(x_train.shape)

lr_tf_idf = OneVsRestClassifier(estimator=LogisticRegression()).fit(x_train, y_train)

preds_valid_tf_idf = lr_tf_idf.predict(x_valid)
#Accuracy
print(recall_score(y_valid, preds_valid_tf_idf, average='weighted'))
#Recall
print(recall_score(y_valid.T, preds_valid_tf_idf.T, average='macro'))
#Precision
print(precision_score(y_valid.T, preds_valid_tf_idf.T, average='macro'))
#F1
print(f1_score(y_valid.T, preds_valid_tf_idf.T, average='macro'))

preds_test_tf_idf = lr_tf_idf.predict(x_test)


#Accuracy
print(recall_score(y_test, preds_test_tf_idf, average='weighted'))
#Recall
print(recall_score(y_test.T, preds_test_tf_idf.T, average='macro'))
#Precision
print(precision_score(y_test.T, preds_test_tf_idf.T, average='macro'))
#F1
print(f1_score(y_test.T, preds_test_tf_idf.T, average='macro'))

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

cm = c_matrix1(y_test, first_occ, preds_test_tf_idf)

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
    return np.array(arr).reshape((4,46))

metrics = c_matrix2(y_test.T, preds_test_tf_idf.T)

print(metrics)

def balanced_accuracy():
    ratio = []
    for i in range(46):
        tpr = (metrics[3][i])/(metrics[2][i]+metrics[3][i]) 
        print(tpr)
        tnr = (metrics[0][i])/(metrics[1][i]+metrics[0][i])
        ratio.append(round((tpr+tnr)/2,2))
    return ratio

print(unique_jobs)

print(np.average(balanced_accuracy()))
#tn, fp, fn, tp

#courses counts
counts_courses = np.sum(preds_test_tf_idf, axis=0)

#which courses appear the most, right or wrong
#positive likelihood ratio = (TP/FN+TP)/(FP/TN+FP)

#% of courses covered by each job title
def coverage(y_test, tp):
    return [round(tp[i]/np.sum(y_test[i]),2) for i in range(len(y_test))]

print(coverage(annots, cm[3]))
