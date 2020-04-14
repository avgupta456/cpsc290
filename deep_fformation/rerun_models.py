import numpy as np

from utils import load_data, train_and_save_model

global_filters = [64, 128]
individual_filters = [16, 64]
combined_filters = [512, 256]
reg= 1e-07
dropout= 0.1

epochs = 600
dataset = "cocktail_party"
fold = 0

test, train, val = load_data("../datasets/" + dataset + "/fold_" + str(0))
train_and_save_model(global_filters, individual_filters, combined_filters,
train, val, test, epochs, dataset, reg=reg, dropout=dropout, fold_num=fold,
no_pointnet=False, symmetric=False)
