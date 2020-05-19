import pandas as pd
import numpy as np

from sklearn import tree
from sklearn import ensemble
from sklearn import metrics


# Read data
def readdata():
    xtrain = pd.read_csv("train.in", header=None).values
    ytrain = pd.read_csv("train.out", header=None).values.reshape(-1,1)
    xtest = pd.read_csv("test.in", header=None).values
    ytest = pd.read_csv("test.out", header=None).values.reshape(-1,1)

    return xtrain[:, :-1], ytrain, xtest[:, :-1], ytest

xtrain, ytrain, xtest, ytest = readdata()

clf = tree.DecisionTreeClassifier(class_weight="balanced")
#clf = ensemble.GradientBoostingClassifier(verbose=1)

clf = clf.fit(xtrain, ytrain)
labels = clf.predict(xtest)

sum = 0
for idt, label in enumerate(labels):
    if label == ytest[idt][0]:
        sum += 1

sum /= len(labels)
print(f"Success rate: {100*sum:2.2f}%")

for idt, label in enumerate(labels):
    if label > 1:
        labels[idt] = 1

ytest = ytest.reshape((len(ytest)))
for idt, label in enumerate(ytest):
    if label > 1:
        ytest[idt] = 1

fpr, tpr, thresholds = metrics.roc_curve(ytest, labels)
tp = tpr[1]
fp = fpr[1]
fn = 1 - tp
tn = 1 - fp

print(f"TP: {100*tp:5.2f}%, FP: {100*fp:5.2f}%\nFN: {100*fn:5.2f}%, TN: {100*tn:5.2f}%")

# DecisionTreeClassifier
# Success rate: 92.23%
# TP: 91.20%, FP:  1.95%
# FN:  8.80%, TN: 98.05%