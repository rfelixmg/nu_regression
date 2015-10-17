# -*- coding: utf-8 -*-
"""
Created on Wed Oct 07 20:44:22 2015

@author: 71351655
"""

# import modules
import numpy as np
from sklearn import linear_model
from sklearn import svm #0.91
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import tree

import csv
from sklearn import preprocessing
from sklearn import cross_validation, metrics
import pickle
import matplotlib.pyplot as plt

# Load estimator

print "Loading model..."
clf = linear_model.BayesianRidge()

def normalization(db_data, nrange = (0,1)):
    
    import numpy as np
    
    db_data = db_data.astype(np.double)
  
    for i in range(db_data.shape[1]):
        
        data = db_data[:, i]
        data = ((data - np.min(data))/ float((np.max(data) - np.min(data))))
        data = (data * (nrange[1] - nrange[0])) + nrange[0]
        
        db_data[:, i] = data.astype(np.double)  
        
    return db_data 
    

def nominal2discrete(col_nominal, u_col = None):
    if u_col is None:
        u_col = np.unique(col_nominal).tolist()
        
    new_nominal = np.zeros(np.shape(col_nominal))    
    for key, p in enumerate(col_nominal):
        new_nominal[key] = u_col.index(p)
    
    return (u_col, new_nominal)
    
    

    
print "Loading database"
"""
" TODO: load database
"""
csvfile = open('data/train.csv', 'rb')
reader = csv.reader(csvfile, delimiter=",")
cdata = list(reader)
train_set = np.array(cdata)
csvfile.close()

print "Transforming database"
#print train_set
'''
# get train_set informations
'''
train_set_header = train_set[0].tolist()
train_set_id = train_set[1::, -2].tolist()
train_set_target = train_set[1::, -1].astype(np.float)
train_set = train_set[1::, :-2:]
(patterns, atts) = np.shape(train_set)

X = np.zeros(np.shape(train_set))
max_min = np.ndarray((atts,2))
att_nominal = []
cols_nominal = []
for key, p in enumerate(train_set.transpose()):
    try:
        p_max = max(p)
        p_min = min(p)
        max_min[key] = np.array([p_max, p_min])
        X.transpose()[key] = p
    except:
        (col_nominal, p) = nominal2discrete(p)
        cols_nominal.append(col_nominal)
        X.transpose()[key] = p
        att_nominal.append(key)
        max_min[key] = np.array([0, 0])

X = X.astype(np.float)
#X = preprocessing.normalize(X, norm='l1')
X = normalization(X)
y = train_set_target

print "Loading Cross Validation configs"
kf = cross_validation.KFold(patterns, n_folds=10)
len(kf)

result = {'mean_absolute_error':[], 
          'mean_squared_error':[], 
          'median_absolute_error':[],
          'r2': [] }
for key, (train_index, test_index) in enumerate(kf):

    print key, "- folds done!" 
    #print("TRAIN:", train_index, "TEST:", test_index)
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    #regr = linear_model.LinearRegression()
    #regr.fit(X_train, y_train)

    y_result = clf.predict(X_test)
    result.get('mean_absolute_error').append(metrics.mean_absolute_error(y_test, y_result))        
    result.get('mean_squared_error').append(metrics.mean_squared_error(y_test, y_result))        
    result.get('median_absolute_error').append(metrics.mean_squared_error(y_test, y_result))        
    result.get('r2').append(metrics.r2_score(y_test, y_result))        

print "-------------------------------"
print "R2: ", np.mean(result.get('r2'))
print "mean_absolute_error: ", np.mean(result.get('mean_absolute_error'))
print "mean_squared_error: ", np.mean(result.get('mean_squared_error'))
print "median_absolute_error: ", np.mean(result.get('median_absolute_error'))

csvfile = open('data/test.csv', 'rb')
reader = csv.reader(csvfile, delimiter=",")
cdata = list(reader)
test_set = np.array(cdata)
csvfile.close()

test_set_header = test_set[0].tolist()
test_set_id = test_set[1::, -1].astype(np.int)
test_set = test_set[1::, :-1:]
(patterns_test, atts_test) = np.shape(test_set)

X_test = test_set
for key, p in enumerate(test_set.transpose()[att_nominal]):
    ( c, X_test.transpose()[att_nominal[key]]) = nominal2discrete(p, cols_nominal[key])

X_test = X_test.astype(np.float)
#X_test = preprocessing.normalize(X_test, norm='l2')
X_test = normalization(X_test)

clf.fit(X, y)
y_predit = clf.predict(X_test)


summary = {'id': test_set_id, 'prediction': y_predit}

mFile = open('C:/summary.csv', 'w')
for key, id in enumerate(summary.get('id')):
    #print "%d; %f;" %(id, summary.get('prediction')[key])
    mFile.write("%d;%f;\n" % (id, summary.get('prediction')[key]))
mFile.close();
#with open('C:/Temp/FELIX/projects/nubank/data/summary.result', 'wb') as fp:
#    pickle.dump(summary, fp)

#with open('C:/Temp/FELIX/projects/nubank/data/10fold.result', 'wb') as fp:
#    pickle.dump(result, fp)
    

"""
" TODO: prepare algorithm
"""

"""
" TODO: prepare result
"""
