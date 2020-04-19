import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras

from process import process_frame
from helper.ds import ds
import constants

def load_model(model_path):
    model = keras.models.load_model(test_model_path, custom_objects={'tf':tf , 'max_people':max_people})
    return model

def predict(X, model):
    preds = model.predict(X)
    return preds

def viz(X_raw, Y_raw, max_people, features, model_path):
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

    preds = predict([X_group, X_pairs], model)
    pred_groups = ds(preds, people)
    truth_groups = ds(Y_new, people)

    pred_groups_dict = {}
    for i in range(people): pred_groups_dict[i+1] = 0
    for i in range(len(pred_groups)):
        for j in range(len(pred_groups[i])):
            pred_groups[i][j] = int(pred_groups[i][j].split("_")[1])
            pred_groups_dict[pred_groups[i][j]] = i+1

    truth_groups_dict = {}
    for i in range(people): truth_groups_dict[i+1] = 0
    for i in range(len(truth_groups)):
        for j in range(len(truth_groups[i])):
            truth_groups[i][j] = int(truth_groups[i][j].split("_")[1])
            truth_groups_dict[truth_groups[i][j]] = i+1

    print(pred_groups)
    print(truth_groups)

    print(pred_groups_dict)
    print(truth_groups_dict)

    time = X_raw[0]
    data = X_raw[1:people*(pre_features+1)+1]
    data = data.reshape(people, pre_features+1)[:,1:].astype('float')
    print(data)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle("Truth vs Pred Groupings, Time = " + str(time))
    axs[0].set_title("Truth Groupings")
    axs[1].set_title("Predicted Groupings")
    
    axs[0].set_xlabel("distance (ft)")
    axs[0].set_ylabel("distance (ft)")
    axs[1].set_xlabel("distance (ft)")
    axs[1].set_ylabel("distance (ft)")

    colors = ['gray', 'blue', 'orange', 'green']

    axs[0].set_xlim(min(data[:,0])-0.5, max(data[:,0])+0.5)
    axs[0].set_ylim(min(data[:,1])-0.5, max(data[:,1])+0.5)

    axs[1].set_xlim(min(data[:,0])-0.5, max(data[:,0])+0.5)
    axs[1].set_ylim(min(data[:,1])-0.5, max(data[:,1])+0.5)

    for i in range(people):
        color = colors[truth_groups_dict[i+1]]
        axs[0].plot(data[i][0], data[i][1], color=color)
        arrow_x = 0.2*np.cos(data[i][2])
        arrow_y = 0.2*np.sin(data[i][2])
        axs[0].arrow(data[i][0], data[i][1], arrow_x, arrow_y,
            head_width=0.1, head_length=0.1, color=color)

    for i in range(people):
        color = colors[pred_groups_dict[i+1]]
        axs[1].plot(data[i][0], data[i][1], color=color)
        arrow_x = 0.2*np.cos(data[i][2])
        arrow_y = 0.2*np.sin(data[i][2])
        axs[1].arrow(data[i][0], data[i][1], arrow_x, arrow_y,
            head_width=0.1, head_length=0.1, color=color)

    plt.show()


raw_path = constants.raw_path
viz_path = constants.viz_path
features = constants.features
max_people = constants.max_people

test_model_path = constants.test_model_path
model = load_model(test_model_path)

X = np.loadtxt(viz_path+"/X.txt", dtype="U50")
Y = np.loadtxt(viz_path+"/Y.txt", dtype="U50")

file1 = open(raw_path + "/groups.txt", "r")
lines1 = file1.readlines()

for loc in range(0, 5):
    viz(X[loc], Y[loc], max_people, features, model)
    print(lines1[loc])
