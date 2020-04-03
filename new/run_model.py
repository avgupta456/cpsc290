import tensorflow as tf
import numpy as np
import pickle
import keras
import math

import f1
from ds import ds

def load_matrix(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def load_data(path):
    train = load_matrix(path + '/train.p')
    test = load_matrix(path + '/test.p')
    val = load_matrix(path + '/val.p')
    return train, test, val

def train_model(model_path, data_path, max_people):
    model = keras.models.load_model(model_path, custom_objects={'tf':tf , 'max_people':max_people})
    model.summary()

    train, test, val = load_data(data_path)
    X, Y, times = val
    num_features = X[0].shape[3]

    preds = model.predict(X)
    print(f1.calc_f1(X, Y, times, preds, 2/3))
    print(f1.calc_f1(X, Y, times, preds, 1))

model_path = "./models/cocktail/model.h5"
max_people = 20

data_path = "./datasets/cocktail/processed/"

train_model(model_path, data_path, max_people)
