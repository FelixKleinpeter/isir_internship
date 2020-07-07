
import numpy as np
import pandas as pd
import random as rd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_regression

from functions import plt_bar

# Random Forest selection
def random_forest(X, y, display=False, max_depth=7):
    clf = RandomForestClassifier(max_depth=max_depth, random_state=0)
    clf.fit(X, y)
    v = X.columns[np.argmax(clf.feature_importances_)]
    if display :
        plt_bar(X, clf)
    return v

# Backward Feature Elimination
def backward_feature_elimination(X, y, display=False): # FIX ========================
    lreg = LinearRegression()
    rfe = RFE(lreg, n_features_to_select=1, step=1)
    rfe = rfe.fit(X, y)

    return X.columns[rfe.support_][0]

# Forward Feature Selection
def forward_feature_selection(X, y, display=False):
    ffs = f_regression(X, y)
    
    return X.columns[np.argmax(ffs[0])]



# Factor Analysis

# Principal Component Analysis

# Independent Component Analysis

#

# Variable Amount (moyenne normalisee de la variable)
def variable_mean_choice(X, y, display = False):
    
    return X.columns[np.argmax([np.mean(X[v]) / (np.max(X[v]) - np.min(X[v])) for v in X.columns])]

# Random Variable Choice
def random_variable_choice(X, y, display = False):
    
    return X.columns[rd.randint(0,X.columns.size-1)]