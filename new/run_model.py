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

#tests a pre-existing model
def test_model(test_model_path, data_path, max_people):
    model = keras.models.load_model(test_model_path, custom_objects={'tf':tf , 'max_people':max_people}) #loads model
    #model.summary()

    train, test, val = utils.load_data(data_path) #loads data
    for data in [train, test, val]:
        X, Y, times = data

        preds = model.predict(X) #gets predictions
        print(calc_f1(X, Y, times, preds, 2/3, 1e-4)) #calculate and print f1 scores
        print(calc_f1(X, Y, times, preds, 1, 1e-4)) #T=1 is the standard threshold
        print()

#loads model location
test_model_path = constants.test_model_path
max_people = constants.max_people

#iterating over folds
for i in range(5):
    data_path = constants.processed_path+"/fold"+str(i)
    print("Fold " + str(i)) #runs above function
    test_model(test_model_path, data_path, max_people)
    print()

#creating a new model
global_filters = [16, 256] #filters applied to group
individual_filters = [32, 32] #filters applied to dyad
combined_filters = [1024, 256] #filters applied to combined output

reg = 1e-7 #more constants
dropout = 0.13 #prevents overfitting
epochs = 200 #epochs to train

#iterating over folds
for i in range(5):
    data_path = constants.processed_path+"/fold"+str(i)
    model_path = constants.model_path+"/fold"+str(i)
    train, test, val = utils.load_data(data_path) #loads data

    #runs main method
    train_model.train_and_save_model(global_filters, individual_filters, combined_filters,
        train, val, test, model_path, epochs, reg, dropout)

end = datetime.datetime.now() #calculates time elapsed
print("Total Time: " + str(end-start)+"\n")
