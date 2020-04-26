import datetime
start = datetime.datetime.now()

import numpy as np
import math
import os

import tensorflow as tf
import keras

from helper.f1 import calc_f1
from helper import utils
import train_model#2 as train_model
import constants


def test_model(test_model_path, data_path, max_people):
    model = keras.models.load_model(test_model_path, custom_objects={'tf':tf , 'max_people':max_people})
    #model.summary()

    train, test, val = utils.load_data(data_path)
    for data in [train, test, val]:
        X, Y, times = data

        preds = model.predict(X)
        print(calc_f1(X, Y, times, preds, 2/3, 1e-4))
        print(calc_f1(X, Y, times, preds, 1, 1e-4))
        print()

test_model_path = constants.test_model_path
max_people = constants.max_people

for i in range(0):
    data_path = constants.processed_path+"/fold"+str(i)
    print("Fold " + str(i))
    test_model(test_model_path, data_path, max_people)
    print()

global_filters = [16, 256]
individual_filters = [32, 32]
combined_filters = [1024, 256]

reg = 1e-7
dropout = 0.13
epochs = 200

for i in range(1):
    data_path = constants.processed_path+"/fold"+str(i)
    model_path = constants.model_path+"/fold"+str(i)
    train, test, val = utils.load_data(data_path)

    train_model.train_and_save_model(global_filters, individual_filters, combined_filters,
        train, val, test, model_path, epochs, reg, dropout)

end = datetime.datetime.now()
print("Total Time: " + str(end-start)+"\n")
