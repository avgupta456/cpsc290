import argparse

from utils import load_data, build_model

import numpy as np
import tensorflow as tf
import pickle
import os

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, Reshape, MaxPooling2D, Concatenate, Lambda, Dot, BatchNormalization, Flatten

from F1_calc import F1_calc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k_fold', type=str, default='0',
        help="the fold being considered")
    parser.add_argument('-m', '--model_path', type=str,
        help="path to the desired model directory (e.g. models/cocktail_party/pair_predictions_1/)", required=True)
    parser.add_argument('-d', '--dataset', type=str, required=True,
        help="which dataset to use (e.g. cocktail_party)")
    parser.add_argument('-f', '--F1', action='store_true', default=False,
        help="calculates the F1 score on the test set, otherwise saves predictions to an output file")
    parser.add_argument('--non_reusable', action='store_true', default=False,
        help="doesn't reuse the same sets in GDSR calc")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    test, train, val = load_data("../datasets/" + args.dataset + "/fold_" + str(args.k_fold))
    X, y, timestamps = test
    num_test, _, max_people, d = X[0].shape

    model = keras.models.load_model(args.model_path + "/val_fold_" + str(args.k_fold)
        + "/best_val_model.h5", custom_objects={'tf':tf , 'max_people':max_people})

    model.summary()
    model.save_weights('weights.h5')

    #taken from architecture file
    #TODO: read these values from file directly
    global_filters = [32, 256, 1024]
    individual_filters = [16, 64]
    combined_filters = [1024, 256]
    reg = 5.011872336272725e-06
    dropout = 0.13

    n_people = 50
    d = 4

    model2 = build_model(reg, dropout, n_people-2, d, global_filters, individual_filters, combined_filters)
    model2.load_weights('weights.h5')
    model2.summary()

    model2.save('new_model.h5')

#Use the below command to run with cocktail_party dataset and pretrained model (fold 2)
#py -3 test.py -k 2 -m models/cocktail_party/pair_predictions_24/ -d cocktail_party
