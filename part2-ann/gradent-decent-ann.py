import six
import sys
sys.modules['sklearn.externals.six'] = six
import csv
import numpy as np 
import pandas as pd 
import mlrose
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime


def main():
    x, y = transformData()
    neuralNet(x,y)
    plotaccuracy(x,y)

def transformData(): 
    data = pd.read_csv('Cancerdata.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['diagnosis','id'], axis=1)
    le = LabelEncoder() 
    y = le.fit_transform(data['diagnosis'])
    return x,y

def neuralNet(x,y):
    clf = mlrose.NeuralNetwork(hidden_nodes = [], activation = 'relu', \
                                    algorithm = 'gradient_descent', max_iters = 1000, \
                                    bias = True, is_classifier = True, learning_rate = 0.0001, \
                                    early_stopping = True, clip_max = 5, max_attempts = 100, \
                                    random_state = 3)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 3)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_test)
    y_train_accuracy = accuracy_score(y_test, y_train_pred)
    print(y_train_accuracy)
    # scores = cross_val_score(clf, x, y, cv=5, verbose=1)
    # print(scores)

def plotaccuracy(x,y):
    accuracy= []
    times=[]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 3)

    for x in range(1,1001,50):
        clf = mlrose.NeuralNetwork(hidden_nodes = [], activation = 'relu', \
                                    algorithm = 'gradient_descent', max_iters = x, \
                                    bias = True, is_classifier = True, learning_rate = 0.0001, \
                                    early_stopping = True, clip_max = 5, \
                                    random_state = 3)
        
       
        start = datetime.datetime.now()   
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        y_test_accuracy = accuracy_score(y_test, y_test_pred)
        stop = datetime.datetime.now()
        accuracy.append(y_test_accuracy*100)
        times.append(((stop - start).microseconds)/ 1000 )
    

    _, axes = plt.subplots(1, 2, figsize=(20, 5))
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    axes[0].grid()
    param_range=np.linspace(1, 1000, len(accuracy))
    axes[0].plot(param_range, accuracy, label="gradient_descent",color="blue", lw=2)
    axes[0].legend(loc="best")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Accuracy score %")

    axes[1].grid()
    param_range=np.linspace(1, 1000, len(times))
    axes[1].plot(param_range, times, label="gradient_descent",color="blue", lw=2)
    axes[1].legend(loc="best")

    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Time in Milliseconds")

    plt.show()

main()