import numpy as np
import math
import os

import tensorflow as tf
import keras

from helper.f1 import calc_f1
from helper import utils
import train_model


def test_model(model_path, max_people):
    model = keras.models.load_model(model_path, custom_objects={'tf':tf , 'max_people':max_people})
    model.summary()

    train, test, val = utils.load_data("cocktail_party")
    for data in [train, test, val]:
        X, Y, times = data
        for i in range(len(times)):
            times[i] = times[i].split(":")[0]+times[i].split(":")[3]
        #print(times)

        new_times = [0]
        for i in range(len(times)):
            if(times[i]!=times[new_times[-1]]):
                new_times.append(i)

        #print(new_times)
        times = new_times

        preds = model.predict(X)
        mse = ((preds - Y)**2).mean(axis=None)
        print(mse)
        print(calc_f1(X, Y, times, preds, 2/3))
        print(calc_f1(X, Y, times, preds, 1))

model_path = "./models/cocktail_party/best_val_model.h5"
max_people = 6

test_model(model_path, max_people)

global_filters = [16, 128, 512]
individual_filters = [32]
combined_filters = [1024, 256, 256]
epochs = 600
reg = 7.943282347242822e-05
dropout = 0.15

train, test, val = load_data("cocktail")
model_path = "./models/cocktail_expanded/model1"

train_and_save_model(global_filters, individual_filters, combined_filters,
    train, val, test, model_path, epochs, reg, dropout)
