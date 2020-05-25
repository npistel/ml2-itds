import pandas as pd
import numpy as np

from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

from scipy.stats import uniform, randint

from matplotlib import pyplot

def score(y, y_pred):
    y = np.copy(y)
    y_pred = np.copy(y_pred)

    for idt, label in enumerate(y_pred):
        if label > 1:
            y_pred[idt] = 1

    y = y.reshape((len(y)))
    for idt, label in enumerate(y):
        if label > 1:
            y[idt] = 1

    c = metrics.confusion_matrix(y, y_pred, normalize="true")

    print(f"TN: {100*c[0][0]:5.2f}%, FP: {100*c[0][1]:5.2f}%\nFN: {100*c[1][0]:5.2f}%, TP: {100*c[1][1]:5.2f}%")

    return c[0][1]
        
# Read data
def readdata():
    xtrain = pd.read_csv("train.in", header=None).values
    ytrain = pd.read_csv("train.out", header=None).values.reshape(-1,1)
    xtest = pd.read_csv("test.in", header=None).values
    ytest = pd.read_csv("test.out", header=None).values.reshape(-1,1)

    return xtrain[:, :-1], ytrain, xtest[:, :-1], ytest

xtrain, ytrain, xtest, ytest = readdata()

#clf = tree.DecisionTreeClassifier(random_state=0, class_weight="balanced")
clf = ensemble.RandomForestClassifier(random_state=0, verbose=1, max_depth=7, max_features=2, max_leaf_nodes=469, min_samples_split=20, n_estimators=38)

clf.fit(xtrain, ytrain)

labels = clf.predict_proba(xtest)
#score(ytest, labels)

# Confusion Matrix for RandomForestClassifier(random_state=0, verbose=1, max_depth=7, max_features=2, max_leaf_nodes=469, min_samples_split=20, n_estimators=38)
# TN: 99.85%, FP:  0.15%
# FN: 10.08%, TP: 89.92%

ytest = ytest.reshape((len(ytest)))

for idt, label in enumerate(ytest):
    if label > 1:
        ytest[idt] = 1

labels2 = []
for idt, label in enumerate(labels):
    labels2.append(1 - label[0])

labels = np.array(labels2)

fpr, tpr, _ = metrics.roc_curve(ytest, labels)
pyplot.plot(fpr, tpr, marker='.', label='Classifier')

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()

exit()

distributions = dict(
    n_estimators = randint(2, 201),
    max_depth = randint(2, 16),
    min_samples_split = randint(2, 21),
    max_features = randint(2, 21),
    max_leaf_nodes = randint(100, 501)
)

x = np.concatenate([xtrain, xtest])
y = np.concatenate([ytrain, ytest])

test_fold = np.concatenate([
    np.full(xtrain.shape[0], -1, dtype=np.int8),
    np.zeros(xtest.shape[0], dtype=np.int8)
])

cv = PredefinedSplit(test_fold)
search = RandomizedSearchCV(clf, distributions, random_state=0, scoring=metrics.make_scorer(score, greater_is_better=False), verbose=2, n_jobs=5, n_iter=30, cv=cv)
search.fit(x, y)
labels = search.predict(xtest)

score(ytest, labels)

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

# DecisionTreeClassifier
# Success rate: 92.23%
# TP: 91.20%, FP:  1.95%
# FN:  8.80%, TN: 98.05%

#c = metrics.confusion_matrix(ytest, labels, normalize="true")
#print(f"TN: {100*c[0][0]:5.2f}%, FP: {100*c[0][1]:5.2f}%\nFN: {100*c[1][0]:5.2f}%, TP: {100*c[1][1]:5.2f}%")