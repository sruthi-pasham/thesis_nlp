from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score, precision_score
from merger import train, valid, test, y_train, y_valid, y_test

vectorizer = TfidfVectorizer(max_features=2000)
x_train = vectorizer.fit_transform(train).toarray()
x_valid = vectorizer.fit_transform(valid).toarray()
x_test = vectorizer.fit_transform(test).toarray()

x_train.shape

lr_td_idf = OneVsRestClassifier(estimator=LogisticRegression()).fit(x_train, y_train)

preds_valid_td_idf = lr_td_idf.predict(x_valid)
#Accuracy
print(recall_score(y_valid, preds_valid_td_idf, average='weighted'))
#Recall
print(recall_score(y_valid.T, preds_valid_td_idf.T, average='macro'))
#Precision
print(precision_score(y_valid.T, preds_valid_td_idf.T, average='macro'))
#F1
print(f1_score(y_valid.T, preds_valid_td_idf.T, average='macro'))

preds_test_td_idf = lr_td_idf.predict(x_test)
#Accuracy
print(recall_score(y_test, preds_test_td_idf, average='weighted'))
#Recall
print(recall_score(y_test.T, preds_test_td_idf.T, average='macro'))
#Precision
print(precision_score(y_test.T, preds_test_td_idf.T, average='macro'))
#F1
print(f1_score(y_test.T, preds_test_td_idf.T, average='macro'))
