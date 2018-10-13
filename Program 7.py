# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 13:45:30 2018

@author: Utkarsh Mishra
"""
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

Insample_score=[]
Out_of_sample_score=[]

# Random forest estimators

for n in [1, 5, 50, 100]:
    forest = RandomForestClassifier(criterion='gini', n_estimators=n, random_state=0, n_jobs=-1)
    forest.fit(X_train, y_train)
    print('N_estimators=%.2d' %n)
    y_test_pred = forest.predict(X_test)
    y_train_pred = forest.predict(X_train)
    a = metrics.accuracy_score(y_train, y_train_pred)
    b = metrics.accuracy_score(y_test, y_test_pred)
    Insample_score.append(a)    
    Out_of_sample_score.append(b)
    print("\nIn-sample score: %.3f,\nOut of sample score: %.3f" % (a,b))
    kfold = StratifiedKFold(n_splits=10, random_state=0).split(X_train, y_train)
    
    
    scores = []
    for k, (train, test) in enumerate(kfold):
        forest.fit(X_train[train], y_train[train])
        score = forest.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: %2d, Acc: %.3f' % (k+1, score))
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    sys.stdout.write(" \n")

# Random forest feature importance    
feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=100,
                                random_state=0)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), 
        importances[indices],color='lightblue',
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

sys.stdout.write(" \n")
print("My name is {Utkarsh Mishra}")
print("My NetID is: {umishra3}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")