
import numpy as np
import pandas as pd
import time

from matplotlib import pyplot as plt

from functions import get_X, get_y, question_from_v, data_without_v, get_movies_scores, remove_empty_variables


def user_questions(data, metric, display = False):
    new_data = data.copy()

    while new_data.item.unique().size > 10 and len(get_X(new_data).columns) > 1:
        X, y = get_X(new_data), get_y(new_data)
        v = metric(X, y, display=display)
        
        avg = np.mean(X[v])
        if avg > 1:
            avg = int(avg)
        
        y_or_n = input(str(v)+"? (y/n)")
        # y_or_n = input(question_from_v(v, threshold=avg))
        if y_or_n == "y" or y_or_n == "Y" or y_or_n == "yes" or y_or_n == "Yes" :
            new_data = data_without_v(new_data, v, avg, lower=False)
        elif y_or_n == "n" or y_or_n == "N" or y_or_n == "no" or y_or_n == "No" :
            new_data = data_without_v(new_data, v, avg, lower=True)
        new_data.drop([v], axis=1, inplace=True)
        
        remove_empty_variables(new_data)
        
    return new_data

def random_questions(data, metric, display = False):
    new_data = data.copy()
    question_count = 0
    
    while new_data.item.unique().size > 10 and len(get_X(new_data).columns) > 1:
        
        X, y = get_X(new_data), get_y(new_data)
        v = metric(X, y, display=display)
        
        avg = np.mean(X[v])
        
        lower = True
        middle = (avg) / (0.0001 + np.max(new_data[v]) - np.min(new_data[v]))
        if np.random.rand() < middle:
            lower = False
        new_data = data_without_v(new_data, v, avg, lower=lower)
        new_data.drop([v], axis=1, inplace=True)
        if display:
            print(v)
            print(avg)
            print(lower)
            print(new_data.item.unique().size)
        
        remove_empty_variables(new_data)
        question_count += 1
        
    return new_data, question_count

def loop_simulation(data, metric, loop=10, display = False):
    amount_found = np.zeros(data.item.unique().size+1) # FIX
    question_counts = []
    
    for k in range(loop):
        data_found, question_count = random_questions(data, metric)
        
        question_counts.append(question_count)
        scores = get_movies_scores(data_found)
        item_found = scores.iloc[:5].item
        
        for i in item_found:
            amount_found[i-1] += 1
        if k % 5 == 0 and display:
            print("k = " + str(k))
    
    return amount_found, np.mean(question_counts)


def metrics_simulations(data, metric_list, metric_names, loopsize):
    times = {}
    results = {}
    question_counts = {}
    
    for metric, name in zip(metric_list, metric_names):
        t = time.time()
        result, question_count = loop_simulation(data, metric, loop=loopsize, display = False)

        times[name] = (time.time() - t) / loopsize
        results[name] = result
        question_counts[name] = question_count
    
    return results, times, question_counts