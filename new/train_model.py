import numpy as np
import math
import os

import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, Reshape, MaxPooling2D
from keras.layers import Concatenate, Lambda, Dot, BatchNormalization, Flatten

from helper.f1 import calc_f1
from helper.ds import ds

class ValLoss(keras.callbacks.Callback):

    def __init__(self, val):
        super(ValLoss, self).__init__()
        self.X, self.Y, self.times = val

        self.best_model = None
        self.best_val_mse = float("inf")
        self.best_epoch = -1

        self.val_f1 = {"f1s": [], "best_f1": float('-inf')}
        self.val_f2_3 = {"f1s": [], "best_f1": float('-inf')}

        self.val_losses = []
        self.train_losses = []

        self.val_mses = []
        self.train_mses = []

    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        if logs['val_mean_squared_error'] < self.best_val_mse:
            self.best_model = self.model
            self.best_val_mse = logs['val_mean_squared_error']
            self.best_epoch = epoch

        f_1 = calc_f1(self.X, self.Y, self.times, self.model.predict(self.X), 1)[2]
        f_2_3 = calc_f1(self.X, self.Y, self.times, self.model.predict(self.X), 2/3)[2]

        for f_1, obj in [(f_1, self.val_f1), (f_2_3, self.val_f2_3)]:
            if f_1 > obj['best_f1']:
                obj['best_f1'] = f_1
                obj['epoch'] = epoch
                obj['model'] = self.model
            obj['f1s'].append(f_1)

        self.val_losses.append(logs['val_loss'])
        self.train_losses.append(logs['loss'])
        self.val_mses.append(logs['val_mean_squared_error'])
        self.train_mses.append(logs['mean_squared_error'])

def write_history(file_name, history, test):
    file = open(file_name, 'w+')

    file.write("best_val: " + str(history.best_val_mse))
    file.write("\nepoch: " + str(history.best_epoch))

    file.write("\nbest_val_f1_1: " + str(history.val_f1['best_f1']))
    file.write("\nepoch: " + str(history.val_f1['epoch']))

    X_test, Y_test, times_test = test
    preds = history.val_f1['model'].predict(X_test)
    p23, r23, f23 = calc_f1(X_test, Y_test, times_test, preds, 2/3)
    p1, r1, f1 = calc_f1(X_test, Y_test, times_test, preds, 1)

    file.write("\ntest_f1s: " + str(f23) + " " + str(f1))
    file.write('\nprecisions: ' + str(p23) + " " + str(p1))
    file.write('\nrecalls: ' + str(r23) + " " + str(r1))

    file.write("\nbest_val_f1_2/3: " + str(history.val_f2_3['best_f1']))
    file.write("\nepoch: " + str(history.val_f2_3['epoch']))

    X_test, Y_test, times_test = test
    preds = history.val_f2_3['model'].predict(X_test)
    p23, r23, f23 = calc_f1(X_test, Y_test, times_test, preds, 2/3)
    p1, r1, f1 = calc_f1(X_test, Y_test, times_test, preds, 1)

    file.write("\ntest_f1s: " + str(f23) + " " + str(f1))
    file.write('\nprecisions: ' + str(p23) + " " + str(p1))
    file.write('\nrecalls: ' + str(r23) + " " + str(r1))

    file.write("\ntrain loss:")
    for loss in history.train_losses:
        file.write('\n' + str(loss))

    file.write("\nval loss:")
    for loss in history.val_losses:
        file.write('\n' + str(loss))

    file.write("\ntrain mse:")
    for loss in history.train_mses:
        file.write('\n' + str(loss))

    file.write("\nval mse:")
    for loss in history.val_mses:
        file.write('\n' + str(loss))

    file.write("\nval 1 f1:")
    for f1 in history.val_f1['f1s']:
        file.write('\n' + str(f1))

    file.write("\nval 2/3 f1:")
    for f1 in history.val_f2_3['f1s']:
        file.write('\n' + str(f1))

    file.close()

def conv(filters, reg, name=None):
    return Conv2D(filters=filters, kernel_size=1, padding='valid', kernel_initializer="he_normal",
        use_bias='True', kernel_regularizer=reg, activation=tf.nn.relu, name=name)

def build_model(reg_amt, drop_amt, max_people, num_features, global_filters, individual_filters, combined_filters):

    group_inputs = keras.layers.Input(shape=(1, max_people, num_features))
    pair_inputs = keras.layers.Input(shape=(1, 2, num_features))

    reg = keras.regularizers.l2(reg_amt)

    y = pair_inputs

    # Dyad Transform
    for filters in individual_filters:
        y = conv(filters, reg)(y)
        y = Dropout(drop_amt)(y)
        y = BatchNormalization()(y)

    y_0 = Lambda(lambda input: tf.slice(input, [0, 0, 0, 0], [-1, -1, 1, -1]))(y)
    y_1 = Lambda(lambda input: tf.slice(input, [0, 0, 1, 0], [-1, -1, 1, -1]))(y)

    '''
    y = MaxPooling2D(name="dyad_pool", pool_size=[1, 2], strides=1, padding='valid')(y)
    y = Dropout(drop_amt)(y)
    y = BatchNormalization()(y)
    y_flat = Flatten()(y)
    '''

    x = group_inputs

    # Context Transform
    for filters in global_filters:
        x = conv(filters, reg)(x)
        x = Dropout(drop_amt)(x)
        x = BatchNormalization()(x)

    x = MaxPooling2D(name="global_pool", pool_size=[1, max_people], strides=1, padding='valid')(x)
    x = Dropout(drop_amt)(x)
    x = BatchNormalization()(x)
    x_flat = Flatten()(x)

    concat = Concatenate(name='concat')([x_flat, Flatten()(y_0), Flatten()(y_1)])

    '''
    concat = Concatenate(name='concat')([x_flat, y_flat])
    '''

    # Final MLP from paper
    for filters in combined_filters:
        concat = Dense(units=filters, use_bias='True', kernel_regularizer=reg, activation=tf.nn.relu,
            kernel_initializer="he_normal")(concat)
        concat = Dropout(drop_amt)(concat)
        concat = BatchNormalization()(concat)

    # final pred
    affinity = Dense(units=1, use_bias="True", kernel_regularizer=reg, activation=tf.nn.sigmoid,
        name='affinity', kernel_initializer="glorot_normal")(concat)

    model = Model(inputs=[group_inputs, pair_inputs], outputs=affinity)

    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False, clipvalue=0.5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['mean_squared_error'])

    return model

# constructs a model, trains it with early stopping based on validation loss, and then
# saves the output to a .txt file.
def train_and_save_model(global_filters, individual_filters, combined_filters,
    train, val, test, model_path, epochs=600, reg=0.0000001, dropout=.35):

    # ensures repeatability
    tf.set_random_seed(0)
    np.random.seed(0)

    num_train, _, max_people, num_features = train[0][0].shape

    # save achitecture
    if not os.path.isdir(model_path): os.makedirs(model_path)
    file = open(model_path + '/architecture.txt', 'w+')
    file.write("global: " + str(global_filters) + "\nindividual: " +
        str(individual_filters) + "\ncombined: " + str(combined_filters) +
        "\nreg= " + str(reg) + "\ndropout= " + str(dropout))

    best_val_mses = []
    best_val_f1 = []
    best_val_f2_3 = []

    X_train, Y_train, times_train = train
    X_test, Y_test, times_test = test
    X_val, Y_val, times_val = val

    # build model
    model = build_model(reg, dropout, max_people, num_features,
        global_filters, individual_filters, combined_filters)

    # train model val_mse
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    history = ValLoss(test) #custom callback implemented above

    #os.system('cls') #hides annoying warnings
    model.fit(X_train, Y_train, epochs=epochs, batch_size=64,
        validation_data=(X_val, Y_val), callbacks=[history, early_stop])

    best_val_mses.append(history.best_val_mse)
    best_val_f1.append(history.val_f1['best_f1'])
    best_val_f2_3.append(history.val_f2_3['best_f1'])

    # save model
    write_history(model_path + '/results.txt', history, test)
    history.val_f1['model'].save(model_path + '/model.h5')

    file.write("\n\nbest overall val loss: " + str(min(best_val_mses)))
    file.write("\n\nbest overall f1 1: " + str(max(best_val_f1)))
    file.write("\n\nbest overall f1 2/3: " + str(max(best_val_f2_3)))
    file.close()
