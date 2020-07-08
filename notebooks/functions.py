

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# VARIABLES

question_dict = {
    'release' : "Do you prefer movies more recent than ",
    'age' : "Are you older than ",
    'gender' : "Are you a man? (y/n)",
    'Action' : "",
    'Adventure' : "",
    'Animation' : "",
    'Children\'s' : "",
    'Comedy' : "",
    'Crime' : "",
    'Documentary' : "",
    'Drama' : "",
    'Fantasy' : "",
    'Film-Noir' : "",
    'Horror' : "",
    'Musical' : "",
    'Mystery' : "",
    'Romance' : "",
    'Sci-Fi' : "",
    'Thriller' : "",
    'War' : "",
    'Western' : "",
}

# FUNCTIONS

    
def question_from_v(variable, threshold=0):
    try:
        if len(question_dict[variable]) == 0:
            return "Do you like " + variable + " movies? (y/n)"
        if threshold <= 1:
            return question_dict[variable]
        return question_dict[variable] + str(threshold) + "? (y/n)"
    except:
        return str(variable) + "? (y/n)" 
    
def data_without_v(data, variable, value, lower=True):
    d = data.copy()
    if lower:
        d = d[d[variable] < value]
    else:
        d = d[d[variable] > value]
    return d

def get_X(data):
    X = data.copy()
    X.drop(['rating','item','user'], axis=1, inplace=True)
    return X

def get_y(data):
    y_ = data.rating.copy()
    y__ = [round(e) for e in y_]
    y = [0 if e < 4 else 1 for e in y__]
    return y

def get_item_names(movies, items):
    names = []
    for i in items:
        names.append(movies.iloc[i-1].title)
    return names

def get_movies_scores(data):
    scores_data = {
        "item" : data.item.unique(),
        "score" : [np.mean(data[data.item == item].rating) for item in data.item.unique()]
    }
    movies_scores = pd.DataFrame(scores_data, columns = ['item','score'])
    movies_scores = movies_scores.sort_values(by="score",ascending=False)
    return movies_scores

def plt_bar(data, clf):
    fig, ax = plt.subplots()
    ax.barh(np.arange(clf.n_features_), clf.feature_importances_, align='center')
    ax.set_yticks(np.arange(clf.n_features_))
    ax.set_yticklabels(data.columns)
    ax.set_xlabel('Performance')

    plt.show()
    
def get_infos(array):
    print("mean : " + str(np.mean(array)) + "\nstd : " + str(np.std(array)) + "\nmax : " + str(np.max(array)))
    
def remove_empty_variables(data):
    for v in get_X(data).columns:
        avg = np.mean(data[v])
        if data[data[v] > avg].size == 0 or data[data[v] <= avg].size == 0 :
            data.drop([v], axis=1, inplace=True)