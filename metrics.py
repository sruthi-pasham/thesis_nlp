import torch
from torch import nn
from numpy import np
from sklearn.metrics import recall_score, f1_score, precision_score


def binary_classification(inputs, targets):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(inputs, targets)

def empty_arr():
    return np.array([]), np.array([]), np.array([]), np.array([])


def metrics(target, preds, acc, recall, f1, precision):
    acc = np.append(acc, recall_score(target, preds, average='weighted'))
    recall = np.append(recall, recall_score(target.T, preds.T, average='macro')) 
    precision = np.append(precision, precision_score(target.T, preds.T, average='macro'))
    f1 = np.append(f1, f1_score(target.T, preds.T, average='macro'))
    return acc, recall, precision, f1