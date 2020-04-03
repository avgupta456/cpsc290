import tensorflow as tf
import numpy as np
import pickle
import keras
import math

from f1 import F1_calc
import ds

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

    for i in range(len(times)-1):
        start = int(times[i])
        end = int(times[i+1])
        points = end - start
        num_people = int(np.sqrt(points))+1

        X_group = X[0][start][0][:num_people-2]
        X_pair = X[1][start][0]
        X_new = np.append(X_pair, X_group)

        Y_truth = Y[start:end]
        Y_pred = preds[start:end]

        print(Y_truth)
        print(Y_pred)
        print(ds.iterate_climb_learned(Y_pred, num_people, num_features))
        print()

    f_2_3, _, _ = F1_calc(2/3, preds, timestamps, groups_at_time, positions,
    n_people, 1e-5, n_features)

    f_1, _, _  = F1_calc(1, preds, timestamps, groups_at_time, positions,
    n_people, 1e-5, n_features)

    return f_2_3, f_1

model_path = "./models/cocktail/model.h5"
max_people = 20

data_path = "./datasets/cocktail/processed/"

train_model(model_path, data_path, max_people)
