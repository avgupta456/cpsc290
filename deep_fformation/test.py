import argparse

from utils import load_data

import numpy as np
import tensorflow as tf
import keras
import os

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
    print(max_people)

    model = keras.models.load_model(args.model_path + "/val_fold_" + str(args.k_fold)
        + "/best_val_model.h5", custom_objects={'tf':tf , 'max_people':max_people})

    model.save_weights('weights.h5', by_name=True)

    print(model)
    print(model.summary())
