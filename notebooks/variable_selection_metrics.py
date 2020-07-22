
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
        #if avg > 1:
        #    avg = int(avg)
        
        y_or_n = input(str(v)+"? (y/n)")
        # y_or_n = input(question_from_v(v, threshold=avg))
        if y_or_n == "y" or y_or_n == "Y" or y_or_n == "yes" or y_or_n == "Yes" :
            new_data = data_without_v(new_data, v, avg, lower=False)
        elif y_or_n == "n" or y_or_n == "N" or y_or_n == "no" or y_or_n == "No" :
            new_data = data_without_v(new_data, v, avg, lower=True)
        
        #remove_empty_variables(new_data)
        
    return new_data

def random_questions(data, metric, display = False, tree = None):
    new_data = data.copy()
    question_count = 0
    t = Tree(-1)
    if tree != None:
        t = tree.copy()
    
    while new_data.item.unique().size > 30 and len(get_X(new_data).columns) > 1:
        
        
        X, y = get_X(new_data), get_y(new_data)
        if t != None and t.v != -1:
            v = t.v
        else:
            v = metric(X, y, display=display)
        
        avg = np.mean(X[v])
        
        lower = True
        middle = (avg) / (0.0001 + np.max(new_data[v]) - np.min(new_data[v]))
        if np.random.rand() < middle:
            lower = False
        new_data_ = data_without_v(new_data, v, avg, lower=lower)
        if new_data_["item"].size == 0:
            return new_data, question_count+1
        else:
            new_data = new_data_
        if t != None and t.v != -1:
            if lower:
                t = t.left
            else:
                t = t.right
        
        if display:
            print(v)
            print(avg)
            print(lower)
            print(new_data.item.unique().size)
        
        question_count += 1
        
    return new_data, question_count

def loop_simulation(data, metric, loop=10, display = False, tree = None):
    amount_found = {i:0 for i in data.item.unique()}
    question_counts = []
    
    for k in range(loop):
        data_found, question_count = random_questions(data, metric, display = False, tree = tree)
        
        question_counts.append(question_count)
        scores = get_movies_scores(data_found)
        item_found = scores.iloc[:5].item
        
        for i in item_found:
            amount_found[i] += 1
        if k % 5 == 0 and display:
            print("k = " + str(k))
    
    return amount_found, question_counts


def metrics_simulations(data, metric_list, metric_names, loopsize, display = False, trees = None):
    times = {}
    results = {}
    question_counts = {}
    
    for metric, name in zip(metric_list, metric_names):
        if display:
            print(" ========== " + name + " ========== ")
        t = time.time()
        if trees == None:
            result, question_count = loop_simulation(data, metric, loop=loopsize, display = display)
        else:
            result, question_count = loop_simulation(data, metric, loop=loopsize, display = display, tree = trees[name])

        times[name] = (time.time() - t) / loopsize
        results[name] = result
        question_counts[name] = question_count
        
    
    return results, times, question_counts

class Tree:
    def __init__(self, v):
        self.v = v
        self.left = None
        self.right = None
    
    def set_left(self, t):
        self.left = t

    def set_right(self, l):
        self.right = l
    
    def copy(self):
        if self.v == -1:
            return Tree(-1)
        else:
            t = Tree(self.v)
            if self.left != None:
                t.set_left(self.left.copy())
            if self.right != None:
                t.set_right(self.right.copy())
            return t

def data_without_sequence(data, sequence):
    """ By default, the threshold of variables are their mean on the initial database """
    new_data = data.copy()
    for (v, b) in sequence:
        X = get_X(new_data)
        avg = np.mean(X[v])
        new_data = data_without_v(new_data, v, avg, lower=(not b))
    return new_data

def pre_compute_tree(data, metric, depth = 5):
    def rec(s):
        new_data = data_without_sequence(data, s)
        if len(s) > depth:
            return Tree(-1)
        if new_data.item.unique().size > 10 and len(get_X(new_data).columns) > 1:
            X, y = get_X(new_data), get_y(new_data)
            v = metric(X, y, display=False)
            
            s1 = s + [(v, True)]
            s2 = s + [(v, False)]
            
            t = Tree(v)
            t.set_left(rec(s1))
            t.set_left(rec(s2))
            
            return t
        return Tree(-1)
    return rec([])