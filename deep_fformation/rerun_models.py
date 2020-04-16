import numpy as np
import tensorflow as tf

from utils import load_data, train_and_save_model

global_filters = [32, 256]
individual_filters = [16]
combined_filters = [512, 128, 64]
reg = 7.943282347242822e-05
dropout = 0.07

epochs = 600
dataset = "cocktail_party"
fold = 1

tf.set_random_seed(0)
np.random.seed(0)

test, train, val = load_data("../datasets/" + dataset + "/fold_" + str(0))
train_and_save_model(global_filters, individual_filters, combined_filters,
train, val, test, epochs, dataset, reg=reg, dropout=dropout, fold_num=fold,
no_pointnet=False, symmetric=False)
