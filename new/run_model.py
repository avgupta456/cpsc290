import numpy as np
import math

import tensorflow as tf
import keras as keras

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, Reshape, MaxPooling2D, Concatenate, Lambda, Dot, BatchNormalization, Flatten

from f1 import calc_f1
from ds import ds

import train_model
import utils

def test_model(model_path, max_people):
    model = keras.models.load_model(model_path+"/model.h5", custom_objects={'tf':tf , 'max_people':max_people})
    model.summary()

    train, test, val = load_data(data_path)
    for data in [train, test, val]:
        X, Y, times = data
        preds = model.predict(X)
        print(calc_f1(X, Y, times, preds, 2/3))
        print(calc_f1(X, Y, times, preds, 1))

model_path = "./models/cocktail/model1"
max_people = 20

#test_model(model_path, data_path, max_people)

global_filters = [32, 256, 1024]
individual_filters = [16, 64]
combined_filters = [1024, 256]
epochs = 600
reg = 5.011872336272725e-06
dropout = 0.13

train, test, val = utils.load_data("cocktail")
train_model.train_and_save_model(global_filters, individual_filters, combined_filters, train, val, test, model_path, epochs, reg, dropout)
