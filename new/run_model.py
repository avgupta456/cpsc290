import datetime
start = datetime.datetime.now()

import numpy as np
import math
import os

import tensorflow as tf
import keras

from helper.f1 import calc_f1
from helper import utils
import train_model
import constants


def test_model(test_model_path, data_path, max_people):
    model = keras.models.load_model(test_model_path, custom_objects={'tf':tf , 'max_people':max_people})
    #model.summary()

    train, test, val = utils.load_data(data_path)
    for data in [train, test, val]:
        X, Y, times = data

        preds = model.predict(X)
        print(calc_f1(X, Y, times, preds, 2/3))
        print(calc_f1(X, Y, times, preds, 1))

test_model_path = constants.test_model_path
max_people = constants.max_people

'''
for i in range(5):
    data_path = constants.processed_path+"/fold"+str(i)
    test_model(test_model_path, data_path, max_people)
    print()
'''

'''
global_filters = [16, 256]
individual_filters = [32, 32]
combined_filters = [1024, 256]

reg = 1e-7
dropout = 0.13
epochs = 200
'''

global_filters = [64, 128]
individual_filters = [16, 64]
combined_filters = [512, 256]

reg= 1e-07
dropout= 0.1
epochs = 600

for i in range(0, 5):
    data_path = constants.processed_path+"/fold"+str(i)
    model_path = constants.model_path+"/fold"+str(i)
    train, test, val = utils.load_data(data_path)

    '''
    X, Y, times = train
    for i in range(len(times)):
        times[i] = times[i].split(":")[0]+times[i].split(":")[3]

    new_times = [0]
    for i in range(len(times)):
        if(times[i]!=times[new_times[-1]]):
            new_times.append(i)

    train = [X, Y, new_times]

    X, Y, times = test
    for i in range(len(times)):
        times[i] = times[i].split(":")[0]+times[i].split(":")[3]

    new_times = [0]
    for i in range(len(times)):
        if(times[i]!=times[new_times[-1]]):
            new_times.append(i)

    test = [X, Y, new_times]

    X, Y, times = val
    for i in range(len(times)):
        times[i] = times[i].split(":")[0]+times[i].split(":")[3]

    new_times = [0]
    for i in range(len(times)):
        if(times[i]!=times[new_times[-1]]):
            new_times.append(i)

    val = [X, Y, new_times]
    '''

    train_model.train_and_save_model(global_filters, individual_filters, combined_filters,
        train, val, test, model_path, epochs, reg, dropout)

end = datetime.datetime.now()
print("Total Time: " + str(end-start)+"\n")
