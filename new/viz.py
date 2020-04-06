import tensorflow as tf
import numpy as np
import keras

from process import process_frame
from helper.ds import ds

def predict(X_raw, Y_raw, max_people, features, model_path):
    pre_features = features[0] + features[1] + features[2]
    post_features = features[0] + 2 * features[1] + features[2]

    X, Y, points = process_frame(X_raw, Y_raw, max_people, features)
    people = int(np.sqrt(points)) + 1

    X_group = np.zeros(shape=(points, 1, max_people-2, post_features))
    X_pairs = np.zeros(shape=(points, 1, 2, post_features))
    Y_new = np.zeros(shape=(points, 1), dtype=np.int8)

    for i in range(points):
        X_group[i][0] = np.reshape(X[i][1:-2*post_features], newshape=(max_people-2, post_features))
        X_pairs[i][0] = np.reshape(X[i][-2*post_features:], newshape=(2, post_features))
        Y_new[i][0] = int(Y[i][1])

    model = keras.models.load_model(model_path+"/model.h5", custom_objects={'tf':tf , 'max_people':max_people})
    preds = model.predict([X_group, X_pairs])

    pred_groups = ds(preds, people)
    truth_groups = ds(Y_new, people)

    print(pred_groups)
    print(truth_groups)

expanded = False
if(expanded): folder = "cocktail_expanded"
else: folder = "cocktail"

raw_path = "./datasets/"+folder+"/raw"
viz_path = "./datasets/"+folder+"/viz"
clean_path = "./datasets/"+folder+"/clean"
processed_path = "./datasets/"+folder+"/processed"
model_path = "./models/"+folder

max_people = 20 #max people possible

#features[0] is space stored for x, y
#features[1] is space stored for angles related to position
#features[2] is space stored for all other features
if(expanded): features = [2, 2, 0]
else: features = [2, 1, 0]

X = np.loadtxt(viz_path+"/X.txt", dtype="U50")
Y = np.loadtxt(viz_path+"/Y.txt", dtype="U50")
predict(X[0], Y[0], max_people, features, model_path)
