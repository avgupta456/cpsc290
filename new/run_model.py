import numpy as np
import math
import os

import tensorflow as tf
import keras

from helper.f1 import calc_f1
from helper import utils
import train_model


def test_model(model_path, max_people):
    model = keras.models.load_model(model_path+"/model.h5", custom_objects={'tf':tf , 'max_people':max_people})
    model.summary()

    train, test, val = utils.load_data("cocktail")
    for data in [train, test, val]:
        X, Y, times = data
        preds = model.predict(X)
        print(calc_f1(X, Y, times, preds, 2/3))
        print(calc_f1(X, Y, times, preds, 1))

model_path = "./models/cocktail"
max_people = 20

test_model(model_path, max_people)

global_filters = [32, 256, 1024]
individual_filters = [16, 64]
combined_filters = [1024, 256]
epochs = 600
reg = 5.011872336272725e-06
dropout = 0.13

train, test, val = utils.load_data("cocktail")
model_path = "./models/cocktail/model1"

#train_model.train_and_save_model(global_filters, individual_filters, combined_filters,
    #train, val, test, model_path, epochs, reg, dropout)
